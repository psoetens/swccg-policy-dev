# How the policy network works, and where it loses card identity

**Purpose:** explain, without assuming machine-learning background, how the policy turns a game state into a choice,
what "block 1" is, and why the measurements in `reports/card-identity-combos.md` point at it. The code is in
`src/swccg_policy/efficient.py` (`EfficientPolicy`, actor and critic encoder), `src/swccg_policy/relational.py`
(`PrintedCards`, `DirectedRound`) and `src/swccg_policy/v4_edges.py`.
**Numbers:** the finals of P-v3, nvf-a and B (2026-10-16). Each model has 256 numbers per token, 6 transformer blocks
with 8 attention heads, and a 1,024-wide feed-forward layer.

## 1. The game state becomes a set of tokens

Every item the policy may need is one **token**: a card in hand, a card at a site, a location, a pile summary, the
question being asked ("Choose Deploy action or Pass"), each option. A token is a list of 256 numbers (a vector). The
Java harness writes the tokens; the network reads them.

A card token's starting vector is the **sum of parts**:

| Part | What it says | Where it comes from |
|---|---|---|
| card | which card this is | the card's printed stats and its text (a shared network, `PrintedCards.prior`) plus a small learned per-card adjustment (`delta`, 22–29% of the size of the printed part) |
| zone | where it is (hand, at a location, attached, …) | a learned vector per zone |
| owner | mine or the opponent's | a learned vector per owner |
| numeric | power, destiny, counts, … | a small layer over the numbers |
| location state / icons | for locations: who controls, icons | small layers |

At this point the network can tell which card a token is almost perfectly: a linear read-back ("probe") of the 200 most
common cards scores 0.99. In the actor the card part is also the largest of these parts (size ~3.5, against ~1.2–3.0
for the others).

## 2. Graph rounds: neighbours exchange information

The game has fixed links: a weapon attached to a character, a character at a site, a pilot aboard a starship. The
harness sends these links as edge tokens, each with a relation type (17 types today). Three **graph rounds**
(`DirectedRound`) pass messages along them: each token absorbs a summary of its directly linked neighbours.

After the graph rounds the card is still readable (probe 0.85–0.88).

## 3. The transformer: six blocks

Now every token may look at every other token, not only at its linked neighbours. This happens in six **blocks**, one
after another. Each block has two sublayers:

- **Attention.** Each token asks "which other tokens matter to me now?", gives every token a weight, and takes in a
  weighted mix of their information. Each block has 8 independent "heads" that can each look for something different.
  This is the only place where "Luke's Lightsaber in my hand" can take into account that "Luke is on the table", so
  combos would have to be recognised here.
- **Feed-forward.** Each token, on its own, passes its vector through a small two-layer network (256 → 1,024 → 256
  numbers). This turns what the token holds into features useful for decisions ("cheap deployable character", "site I
  control").

**Blocks add to a token; they do not replace it.** Picture each token as a page of notes. A block first normalises the
page to a standard size and reads it. It then *writes additional notes* on the page: the attention result, then the
feed-forward result. The running page is called the *residual stream*.

```
token page ──► block 1: + attention notes + feed-forward notes
           ──► block 2: + notes
           ...
           ──► block 6: + notes ──► final normalisation ──► final page
```

"Block 1" is simply the first of the six.

## 4. Scoring the options

For an option such as "deploy card X", the network combines:
- X's own card vector, taken directly from the card table (it does not go through the blocks);
- the final page of X's token on the board (its *source token*);
- a pooled view of the option's rows and the question.

A small layer turns this into a score per option. The policy then picks the highest score (greedy evaluation) or
samples by score (training rollouts).

The critic, which estimates the chance of winning to guide training, has its own copy of steps 1–3 and reads the
final pages.

## 5. What goes wrong in block 1

Measured on card tokens (`reports/card-identity-combos.md` M3b):

| Actor | Page size entering block 1 | Notes block 1 adds | Of which feed-forward | Change per later block (blocks 3–6) |
|---|---|---|---|---|
| P-v3 | 18 | 707 | 702 | 25–51 on a page of ~800 |
| nvf-a | 19 | 794 | 788 | 44–54 on ~900 |
| B | 20 | 953 | 948 | 56–62 on ~1,070 |

- **Block 1's feed-forward writes notes 35–50× larger than the page it received.** The original content, including
  which card this is, is still on the page but is now only ~2–3% of it.
- **Every later block reads a normalised page,** so to them the original content is faint. Their own notes are small:
  each changes the page by 3–6%, attention included.
- **The network therefore works as "graph rounds, one big per-token rewrite, small corrections".** What one card learns
  about another, which is what a combo needs, is one of those small corrections.
- **The card read-back confirms it:** 0.99 at the input, 0.85–0.88 after the graph rounds, **0.47–0.51 after block 1**,
  and flat from there (majority guess: 0.34–0.37).
- **The critic is far more extreme:** block 1 writes 600–1,000× its input, and from block 1 on the card cannot be read
  back at all (the probe equals its majority guess, 0.57–0.62).

## 6. Why training ends up there

Each block normalises what it *reads*, not what it *writes*. Nothing limits the size of the notes. If large, generic
notes help with what training currently rewards, training makes them large:
- drains and battles decide 59–62% and 32–34% of wins against AdvancedAI;
- features like "deployable character" or "contested site" serve them well.

Card identity and card-to-card relations get drowned as a side effect. Nothing pushes back, because combos almost never
pay off in training (card effects decide 5–8% of wins).

## 7. Why decisions still depend on the card

The option score receives the played card's vector directly, bypassing the blocks (step 4).
- Swapping the identity of the card being decided on changes 30–35% of decisions.
- Swapping another card on the board changes 3–4%, and its zone about 1%.

The policy knows *what* it is playing, but sees the board around it mostly as generic features. It does not connect a
card to the cards it combines with:
- The payoff's token attends to its enabler no more than to any other own card.
- Removing the enabler does not change the payoff's score.
- The policy plays payoff cards with their enabler in play at 0.44–0.51× the chance rate (AdvancedAI: ~1.0).

## 8. The two fixes

- **Fix 1: bound what a block writes.** Each sublayer's output is normalised and scaled before it is added, so no
  single block can drown the page. The original content (which card) stays readable, and attention's notes, the
  card-to-card information, keep real weight. It applies to the actor and the critic.
- **Fix 2: "names" links before block 1.** A message step at the end of the graph rounds links every card token to the
  own cards its text names: the 1,650 within-deck pairs of the combo map, 84% precise. Identity is still intact there
  (0.85–0.88), so the payoff's token receives its enabler's state directly, instead of relying on attention to discover
  1,650 rare relations from rare rewards.

The A/B plan that tests them is `.claude/plans/dilution-names-ab.md`.
