"""Import four Decipher-era Carbon Chamber Testing decks with Jabba's Prize (user 2026-10-04: CCT+JP round-2 pool).

Lists were taken from the web (sources below) and written out as canonical card titles; nicknames were translated by
hand (e.g. "DVDLOTS" -> Darth Vader, Dark Lord Of The Sith; "Iggy In Boat" -> IG-88 In IG-2000). Titles are resolved
against n1's card table (Dark Side only); every title must resolve to exactly one Dark Side card, else the script stops.
Writes data/decks/v4/corpus/web-cctjp-<name>.xml and prints a resolution report.
Not imported: "Frustration" (hanudel-sabacc) has no Decipher Dark Side card of that name; the deck has 59 cards.
"""
import collections
from xml.sax.saxutils import quoteattr
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/decks/v4/corpus'

DECKS = {
    'oakes-unh-elimination': ('http://www.theforce.net/ccg/sw/decks/cctnf.asp (Jeremy "JerUNH" Oakes)', """
1 Carbon Chamber Testing
1 Carida
1 Cloud City: Carbonite Chamber
1 Cloud City: Casino
1 Cloud City: Downtown Plaza
1 Cloud City: East Platform (Docking Bay)
1 Cloud City: Security Tower
1 4-LOM With Concussion Rifle
1 Admiral Chiraneau
1 Admiral Piett
1 Boba Fett With Blaster Rifle
1 Commander Igar
1 Dengar With Blaster Carbine
2 Emperor Palpatine
1 Grand Admiral Thrawn
1 Grand Moff Tarkin
1 Guri
1 IG-88 With Riot Gun
1 Jabba's Prize
1 Janus Greejatus
1 Lando Calrissian
2 Lord Vader
1 Mara Jade, The Emperor's Hand
2 Mercenary Pilot
1 Prince Xizor
1 Sim Aloo
1 Snoova
1 Blizzard 2
1 Lift Tube
1 Patrol Craft
1 Tempest 1
1 Tempest Scout 5
1 Bossk In Hound's Tooth
1 Chimaera
1 Executor
1 Zuckuss In Mist Hunter
1 Mara Jade's Lightsaber
1 Vader's Lightsaber
1 Vibro-Ax
1 Carbonite Chamber Console
1 Mobilization Points
2 Presence Of The Force
1 There Is No Try & Oppressive Enforcement
2 There'll Be Hell To Pay
1 Focused Attack
2 Force Lightning
2 Imperial Barrier
1 Prepared Defenses
1 The Circle Is Now Complete
2 Trample
1 Vader's Obsession
1 Weapon Levitation
"""),
    'ford-dark-deal': ('https://www.stephenskilton.com/decktech_archives/15317/ (Doug "Froggy" Ford, 2001-04-16)', """
1 Carbon Chamber Testing
1 Cloud City: Security Tower
1 Cloud City: Carbonite Chamber
1 Jabba's Prize
1 Carbonite Chamber Console
1 Prepared Defenses
1 All Wrapped Up
1 Inconsequential Losses
1 Mobilization Points
2 Bespin: Cloud City
1 Jabba's Palace: Audience Chamber
1 Tatooine: Docking Bay 94
1 Cloud City: East Platform (Docking Bay)
1 Cloud City: Interrogation Room
1 Cloud City: Upper Plaza Corridor
2 Twi'lek Advisor
2 Sniper
3 None Shall Pass
2 Jabba's Through With You
1 Reactor Terminal
1 Bad Feeling Have I
1 Insignificant Rebellion
1 Dark Deal
2 Bounty
4 Intruder Missile
1 Vader's Lightsaber
1 Mara Jade's Lightsaber
1 4-LOM's Concussion Rifle
1 Dengar's Modified Riot Gun
1 Feltipern Trevagg's Stun Rifle
1 Blaster Rifle
1 Executor
2 Imperial-Class Star Destroyer
1 OS-72-1 In Obsidian 1
1 OS-72-2 In Obsidian 2
1 Obsidian 7
1 Obsidian 8
2 Darth Vader, Dark Lord Of The Sith
2 Mara Jade, The Emperor's Hand
2 Grand Moff Tarkin
1 4-LOM
1 Dengar
1 Feltipern Trevagg
1 Jabba The Hutt
1 Lando Calrissian
1 Bane Malar
"""),
    'wilcox-bastardized': ('https://www.stephenskilton.com/decktech_archives/14344/ (Ronnie "AD-fig" Wilcox, 2001-03-07)', """
1 Carbon Chamber Testing
1 Cloud City: Carbonite Chamber
1 Cloud City: Security Tower
1 Prepared Defenses
1 Crush The Rebellion
1 Mobilization Points
1 Imperial Arrest Order
1 Carbonite Chamber Console
1 Jabba's Prize
1 Tatooine: Docking Bay 94
1 Cloud City: East Platform (Docking Bay)
1 Jabba's Palace: Audience Chamber
1 Carida
1 Mighty Jabba
1 Darth Vader With Lightsaber
2 Darth Vader, Dark Lord Of The Sith
1 Lord Vader
1 Mara Jade, The Emperor's Hand
1 Dr. Evazan & Ponda Baba
1 4-LOM With Concussion Rifle
1 Emperor Palpatine
1 Sim Aloo
1 Barquin D'an
1 IG-88 With Riot Gun
2 Janus Greejatus
1 Prince Xizor
1 Guri
1 Search And Destroy
1 Something Special Planned For Them
1 Battle Order|8_118
1 First Strike
1 You Cannot Hide Forever|9_133
1 Darth Vader's Lightsaber
1 Vader's Lightsaber
1 Mara Jade's Lightsaber
2 Vader's Obsession
1 Twi'lek Advisor
1 Shut Him Up Or Shut Him Down
1 Operational As Planned
1 The Circle Is Now Complete
1 Focused Attack
1 Holonet Transmission
1 This Is Some Rescue!
2 I Have You Now
1 Evader & Monnok
1 Sniper & Dark Strike
1 Ommni Box & It's Worse
1 Masterful Move
1 Scanning Crew
2 Imperial Barrier
2 Bossk In Hound's Tooth
1 Zuckuss In Mist Hunter
1 IG-88 In IG-2000
"""),
    'hanudel-sabacc': ('https://www.stephenskilton.com/decktech_archives/12770/ (Christopher "Thrawn" Hanudel, 2001-01-08)', """
1 Carbon Chamber Testing
1 Cloud City: Carbonite Chamber
1 Cloud City: Security Tower
2 Cloud City: Casino
1 Jabba's Palace: Audience Chamber
1 Cloud City: Port Town District
1 Darth Vader
1 Lord Vader
1 Darth Vader With Lightsaber
1 Grand Moff Tarkin
1 Jabba's Prize
2 Lando Calrissian
2 Mara Jade, The Emperor's Hand
2 Dr. Evazan
2 4-LOM With Concussion Rifle
1 IG-88 With Riot Gun
1 Dengar With Blaster Carbine
2 Corporal Prescott
1 Sim Aloo
1 Ephant Mon
1 U-3PO (Yoo-Threepio)
6 Cloud City Sabacc
2 Trooper Sabacc
3 Twi'lek Advisor
2 Imperial Barrier
1 Hutt Smooch
1 Sniper
1 Prepared Defenses
2 Disarmed
2 Expand The Empire
1 Power Of The Hutt
1 Reactor Terminal
1 Crush The Rebellion
1 There Is No Try & Oppressive Enforcement
1 Imperial Arrest Order
1 Security Precautions
1 Carbonite Chamber Console
1 Mara Jade's Lightsaber
1 Vader's Lightsaber
1 Boba Fett In Slave I
1 Bossk In Hound's Tooth
1 Zuckuss In Mist Hunter
"""),
}


def main():
    cards = torch.load(ROOT / 'runs/knowledge-v3.1/v31-train8h-n1/final.pt', map_location='cpu',
                       weights_only=False)['tables']['cards']
    by_title = collections.defaultdict(list)
    for blueprint, title, side in zip(cards['ids'], cards['titles'], cards['sides']):
        by_title[str(title).lower()].append((str(blueprint), str(title), str(side)))
    problems = []
    for name, (source, text) in DECKS.items():
        rows = []
        for line in text.strip().splitlines():
            count, title = line.split(' ', 1)
            title, _, pinned = title.partition('|')  # 'Title|id' picks one printing (original, not the reprint)
            hits = [h for h in by_title.get(title.lower(), []) if h[2] == 'DARK' and (not pinned or h[0] == pinned)]
            if len(hits) != 1:
                problems.append(f'{name}: {title!r} -> {hits or "no Dark Side match"}')
                continue
            rows += [hits[0]] * int(count)
        ids = [r[0] for r in rows]
        print(f'{name}: {len(rows)} cards, CCT={ids.count("7_296")}, Jabba\'s Prize={ids.count("10_42")}  [{source}]')
        if not problems:
            xml = "<?xml version='1.0' encoding='UTF-8'?>\n<deck>\n" + ''.join(
                f'  <card blueprintId="{b}" title={quoteattr(t)} />\n'  for b, t, _ in rows) + '</deck>\n'
            (OUT / f'web-cctjp-{name}.xml').write_text(xml)
    for p in problems:
        print('UNRESOLVED', p)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
