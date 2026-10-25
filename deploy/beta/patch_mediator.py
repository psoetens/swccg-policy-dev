"""Plan beta-server Amendment 3: no decision ("request") timer in games against a bot (Peter). Patches the build
copy of Gemp's SwccgGameMediator (never the fork): with -Dbeta.noDecisionTimerVsBot=true (entrypoint.sh), a game with
an AI participant (AiRegistry.isAi) never ends on a decision timeout. Games between humans keep the timer; the
per-player game clock and the 3-minute no-action cancel are unchanged.

    python deploy/beta/patch_mediator.py BUILD_DIR
"""
import sys
from pathlib import Path

FIELD_OLD = '    private boolean _disablePlayerDecisionTimer;\n'
FIELD_NEW = (FIELD_OLD + '    // Plan beta-server (deploy/beta/patch_mediator.py): no decision timer in games against a bot.\n'
             '    private static final boolean BETA_NO_TIMER_VS_BOT = Boolean.getBoolean("beta.noDecisionTimerVsBot");\n')
CHECK_OLD = 'else if (!_disablePlayerDecisionTimer && (currentTime > decisionSent + _playerDecisionTimeoutPeriod)) {'
CHECK_NEW = ('else if (!_disablePlayerDecisionTimer && !(BETA_NO_TIMER_VS_BOT && _playerClocks.keySet().stream()'
             '.anyMatch(p -> AiRegistry.isAi(_gameId, p))) && (currentTime > decisionSent + _playerDecisionTimeoutPeriod)) {')


def main():
    path = Path(sys.argv[1]) / 'gemp-swccg-server/src/main/java/com/gempukku/swccgo/game/SwccgGameMediator.java'
    text = path.read_text(encoding='utf-8')
    for old, new in ((FIELD_OLD, FIELD_NEW), (CHECK_OLD, CHECK_NEW)):
        if text.count(old) != 1:
            raise SystemExit(f'{path}: expected one occurrence of {old.strip()!r}, found {text.count(old)}')
        text = text.replace(old, new)
    path.write_text(text, encoding='utf-8')
    print(f'{path.name}: no decision timer against bots (beta.noDecisionTimerVsBot)')


if __name__ == '__main__':
    main()
