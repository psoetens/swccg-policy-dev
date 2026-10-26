"""Plan beta-server: mark every page of the staged Gemp web folder as the unofficial beta server.

    python deploy/beta/banner.py WEB_DIR "dev <sha> · gemp <sha>"

Every *.html page gets a fixed corner ribbon (clicks pass through it) with the version on hover; the login page also
gets a notice bar fixed at the bottom. Removed: Google Tag Manager (the official server's analytics) and the official
contact addresses (ban appeals, password resets go to the beta admin). The page title gets a
"[BETA]" prefix. Idempotent per staging: build.sh copies a fresh web folder each time.
"""
import re
import sys
from pathlib import Path

MARK = 'brainiac-beta-banner'
NOTICE = ('UNOFFICIAL BETA TEST SERVER for the Brainiac bot. This is not gemp.starwarsccg.org: accounts, decks and '
          'games may be wiped at any time. Passwords are stored without salt, so use one you use nowhere else. '
          'Browser console logs are sent to this server for testing (add ?debug=0 to a page address to stop). '
          'Also on this server: the <a href="/GempArena/" style="color: #fff">GempArena</a> client (beta).')
STYLE = f"""<style id="{MARK}-style">
#{MARK}-corner {{position: fixed; top: 0; right: 0; width: 90px; height: 90px; overflow: hidden;
  z-index: 2147483647; pointer-events: none;}}
#{MARK}-ribbon {{position: absolute; top: 16px; right: -40px; transform: rotate(45deg); background: #c62828;
  color: #fff; font: bold 12px/20px sans-serif; letter-spacing: 2px; width: 150px; text-align: center;
  box-shadow: 0 1px 4px rgba(0, 0, 0, .5); opacity: .9;}}
#{MARK}-notice {{position: fixed; left: 0; right: 0; bottom: 0; z-index: 2147483646; background: #c62828;
  color: #fff; font: 14px/1.4 sans-serif; padding: 6px 100px; text-align: center;}}
</style>"""
# The official server's contacts (ban appeals, password resets) do not apply here.
CONTACTS = [(re.compile(r"contact the Communications Advocate at <a href='mailto:communications@starwarsccg\.org[^']*'>"
                        r"communications@starwarsccg\.org</a>"), 'contact the beta server admin'),
            (re.compile(r"<a href='mailto:communications@starwarsccg\.org\?subject=Gemp%20Password%20Reset'>Forgot your "
                        r"password\? Contact the <span style='color:orange'>Communications Advocate</span></a>\."),
             'Forgot your password? Ask the beta server admin.')]
CLIENTLOG = '<script src="/clientlog.js" data-client="classic"></script>'
# Peter 2026-10-26: the hall's "Public Beta: New UI" link opens this server's GempArena (/GempArena/)
NEW_UI = ('href="https://gemp.starwarsccg.org/gemp-swccg/epic-duel/newgui.html"', 'href="/GempArena/"')
GTM = re.compile(r'\s*<script async src="https://www\.googletagmanager\.com/gtag/js[^"]*"></script>\s*'
                 r'<script>\s*window\.dataLayer.*?</script>', re.S)


def mark(page: Path, version: str) -> None:
    html = page.read_text(encoding='utf-8')
    html, removed = GTM.subn('\n', html)
    if 'googletagmanager' in html:
        raise SystemExit(f'{page}: Google Tag Manager left after removal')
    for pattern, text in CONTACTS:
        html = pattern.sub(text, html)
    if 'communications@starwarsccg.org' in html:
        raise SystemExit(f'{page}: an official contact address is left')
    if page.name == 'hall.html':
        if NEW_UI[0] not in html:
            raise SystemExit(f'{page}: the "Public Beta: New UI" link was not found')
        html = html.replace(*NEW_UI)
    ribbon = f'<div id="{MARK}-corner"><div id="{MARK}-ribbon" title="{version}">BETA</div></div>'
    notice = f'<div id="{MARK}-notice">{NOTICE} <small>({version})</small></div>' if page.name == 'index.html' else ''
    html = re.sub(r'<title>', '<title>[BETA] ', html, count=1, flags=re.I)
    # Plan client-logs: the console goes to the server; loaded first so it sees the page's own start-up
    head = re.search(r'<head[^>]*>', html, re.I)
    if head:
        html = html[:head.end()] + '\n' + CLIENTLOG + html[head.end():]
    else:
        html = CLIENTLOG + html
    if re.search(r'</head>', html, re.I):
        html = re.sub(r'</head>', STYLE + '\n</head>', html, count=1, flags=re.I)
    else:
        html = STYLE + html
    body = re.search(r'<body[^>]*>', html, re.I)
    insert = ribbon + notice
    html = html[:body.end()] + insert + html[body.end():] if body else html + insert
    page.write_text(html, encoding='utf-8')
    print(f'{page.name}: banner{" + notice" if notice else ""}{", analytics removed" if removed else ""}')


def main():
    web, version = Path(sys.argv[1]), sys.argv[2]
    pages = sorted(web.glob('*.html'))
    if not pages:
        raise SystemExit(f'no html pages in {web}')
    for page in pages:
        mark(page, version)


if __name__ == '__main__':
    main()
