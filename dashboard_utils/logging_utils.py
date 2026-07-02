from datetime import datetime

_LOG_ICONS = {
    "info":   ("💬", "#58a6ff"),
    "ok":     ("✅", "#3fb950"),
    "warn":   ("⚠️",  "#d29922"),
    "error":  ("❌", "#f85149"),
    "start":  ("🚀", "#bc8cff"),
    "dl":     ("⬇️",  "#79c0ff"),
    "map":    ("🗺️",  "#56d364"),
    "run":    ("⚙️",  "#e3b341"),
    "done":   ("🏁", "#3fb950"),
    "search": ("🔍", "#79c0ff"),
}

# Set by the notebook: _log_out_ref[0] = log_output_widget (HTML widget)
_log_out_ref = [None]
_log_lines   = []
_MAX_LINES   = 150  # keep small — large HTML strings slow down the widget

# Anchor div — browser keeps scroll at bottom via CSS overflow-anchor
class WidgetStream:
    def __init__(self, out): self.out = out
    def write(self, text):
        log(text.rstrip(), 'info')
    def flush(self): pass


def _get_log_out():
    return _log_out_ref[0]


def log(msg, kind="info"):
    lo = _get_log_out()
    if lo is None:
        return
    icon, colour = _LOG_ICONS.get(kind, ("💬", "#cdd9e5"))
    ts = datetime.now().strftime("%H:%M:%S")
    line = (
        f'<div style="font-family:monospace;font-size:12px;'
        f'padding:2px 6px;border-bottom:1px solid #21262d;line-height:1.6">'
        f'<span style="color:#484f58">{ts}</span> '
        f'{icon} <span style="color:{colour}">{msg}</span></div>'
    )
    # Newest line at top — no scroll management needed, latest always visible
    _log_lines.insert(0, line)
    if len(_log_lines) > _MAX_LINES:
        del _log_lines[_MAX_LINES:]
    lo.value = ''.join(_log_lines)


def clear_log():
    _log_lines.clear()
    lo = _get_log_out()
    if lo is not None:
        lo.value = ''
