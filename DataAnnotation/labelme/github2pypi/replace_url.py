import re


def replace_url(slug, content, branch='master'):

    def repl(match):
        if not match:
            return

        url = match.group(1)
        if url.startswith('http'):
            return match.group(0)

        url_new = (
            'https://github.com/{slug}/blob/{branch}/{url}'
            .format(slug=slug, branch=branch, url=url)
        )
        if re.match(r'.*[\.jpg|\.png]$', url_new):
            url_new += '?raw=true'

        start0, end0 = match.regs[0]
        start, end = match.regs[1]
        start -= start0
        end -= start0

        res = match.group(0)
        res = res[:start] + url_new + res[end:]
        return res

    lines = []
    for line in content.splitlines():
        patterns = [
            r'!\[.*?\]\((.*?)\)',
            r'<img.*?src="(.*?)".*?>',
            r'\[.*?\]\((.*?)\)',
            r'<a.*?href="(.*?)".*?>',
        ]
        for pattern in patterns:
            line = re.sub(pattern, repl, line)
        lines.append(line)
    return '\n'.join(lines)
