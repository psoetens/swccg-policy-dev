# Brainiac beta server

Plan: `beta-server` (lab `.claude/plans/beta-server.md`, approved 2026-10-25, Amendments 1–3). Status: **live**; Peter
played on it ("Game play works", 2026-10-25). Release updated 2026-10-25 15:30 to the first V-card-trained model.

**https://brainiac-beta.duckdns.org/** — Gemp `feat/brainiac` + the Brainiac overlay, serving
`v-cards-vct-2026-10-25` (`models/vcards-vct`; before: `v-cards-shc095-2026-10-23`); image
`dev-3bba72e2d-gemp-28a27be02` (pushed commits only). Release update checks: `data/beta-server/live-check-vct.txt`.

## Setup

| Part | What |
|---|---|
| Host | Hetzner Cloud, Helsinki, Docker CE app image (Ubuntu 24.04.1, Docker 27.5.1), 2 vCPU, 3.7 GB RAM, 38 GB disk; IPv6 only (`2a01:4f9:c014:53f4::1`) |
| Name | `brainiac-beta.duckdns.org` (AAAA only; no IPv4 record) |
| Firewall | Hetzner cloud firewall: 22, 80, 443 in. SSH: keys only (`deploy.sh harden`) |
| Containers | `caddy` (HTTPS, Let's Encrypt, host network), `gemp` (127.0.0.1:17101, heap 2.5 GB), `db` (MariaDB, no port) |
| Data | `/opt/brainiac`: `.env` (passwords made on the server, root-only), `brainiac/releases` + `brainiac/runtime`, `backups/` (nightly dump 04:00, 14 kept) |
| Browser side | the policy runs in the player's browser (onnxruntime-web); a new player downloads the release once (~76 MB) |

What the beta changes compared with stock Gemp (all at build time, the Gemp fork is untouched):
- every page: "BETA" corner ribbon, "[BETA]" title; login page: notice bar (unofficial, may be wiped, unsalted
  passwords: use a unique one). Google Tag Manager and the official contact addresses removed;
- database: Gemp's seeded users removed (Librarian kept, random password); bot tables on (Gemp's default is off);
  registration on; library = the official server's 90 sample decks (identical names and contents) + the 42 decks of
  the Brainiac test rig (132);
- an ops admin (`betaops`, random password in `.env`) sets the hall operational after every start;
- no decision timer in games against a bot (`patch_mediator.py`, `-Dbeta.noDecisionTimerVsBot`); human games keep it.

## Admin how-to (from the dev repo, on this PC)

```bash
deploy/beta/deploy.sh status
deploy/beta/deploy.sh promote <account>          # make a registered account admin
deploy/beta/deploy.sh push models/brainiac/releases/<release>   # new commit/release: build, copy
deploy/beta/deploy.sh up <release>                # restart on it
deploy/beta/check.sh https://brainiac-beta.duckdns.org models/brainiac/releases/<release>
```
Registration switch, bans, message of the day: Gemp's admin pages (Admin → settings) with an admin account.

## §1 end-goal

> 1. Local rehearsal … check passes … banner in every served HTML page … seeded users cannot log in … admin can
>    toggle registration … one full game vs Brainiac played to the end in a real browser by Peter.
> 2. Live: `https://<name>.duckdns.org/gemp-swccg/` serves with a valid certificate; from this PC only ports 22, 80,
>    443 answer; the same checks as 1 pass against the live URL; SSH accepts keys only.
> 3. Peter signs off after one game on the live server.

1. Rehearsal: `data/beta-server/rehearsal-check.txt` — all checks pass. Peter's first browser try found the missing
   runtime (fixed, Amendment 2); his full game was then played on the live server (3).
2. Live: `data/beta-server/live-check.txt` — all checks pass over HTTPS (valid Let's Encrypt certificate); ports 22,
   80, 443 open, 3306 / 8000 / 8080 / 17101 / 35001 closed; listeners other than 22/80/443 on localhost only;
   `passwordauthentication no`, password login refused. Screenshot `data/beta-server/live-login.png`.
3. Peter: "Game play works" (live). Admin account `peter` promoted (`deploy.sh promote peter`: type `ua`).

## Skipped / unverified

- Not verified: a bot game left idle > 5 min (no timeout expected), a human game timing out; restore from a backup;
  load beyond a few players.
- Testers without IPv6 cannot reach the site (an IPv4 address on the server + the DuckDNS ipv4 field fixes it).
- The live check's two test accounts (`t92934247`, `y92934247`) were deleted on Peter's word, as were those of
  the release-update check (`t92937882`, `y92937882`); accounts now: Librarian, betaops, peter. `check.sh` makes two
  test accounts on each run; delete them afterwards.
- A crawler found the site within seconds of the certificate (Let's Encrypt logs are public): expect probes.
