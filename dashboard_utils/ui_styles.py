"""
HBV Dashboard — design system v3  (dark theme)
"""
import ipywidgets as widgets
from IPython.display import display, HTML as _HTML

# ── CSS ────────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ══════════════════════════════════════════════════════════════════════════
   GLOBAL DARK THEME — targets both JupyterLab and classic Jupyter Notebook
   ══════════════════════════════════════════════════════════════════════════ */

/* Page / body */
body,
.jp-NotebookPanel,
.jp-Notebook,
#notebook,
.notebook_app,
.container-fluid,
div#notebook-container {
  background: #020617 !important;
  color: #e2e8f0 !important;
}

/* Cell wrappers */
.jp-Cell,
.jp-CodeCell,
.jp-MarkdownCell,
.cell,
div.cell {
  background: #020617 !important;
  border-color: #1e293b !important;
}

/* Cell inputs (code area) */
.jp-InputArea,
.jp-InputArea-editor,
.jp-CodeMirrorEditor,
.CodeMirror,
.CodeMirror-scroll,
.input_area,
div.input_area {
  background: #0f172a !important;
  color: #e2e8f0 !important;
  border-color: #1e293b !important;
}

/* CodeMirror token colours  */
.CodeMirror { color: #e2e8f0 !important; }
.cm-s-default .cm-keyword  { color: #c084fc !important; }
.cm-s-default .cm-string   { color: #4ade80 !important; }
.cm-s-default .cm-number   { color: #fb923c !important; }
.cm-s-default .cm-comment  { color: #475569 !important; font-style: italic; }
.cm-s-default .cm-def      { color: #60a5fa !important; }
.cm-s-default .cm-variable { color: #e2e8f0 !important; }
.cm-s-default .cm-operator { color: #94a3b8 !important; }
.CodeMirror-cursor { border-left-color: #e2e8f0 !important; }
.CodeMirror-selected, .CodeMirror-focused .CodeMirror-selected {
  background: #1e3a5f !important;
}

/* Cell output wrapper */
.jp-OutputArea,
.jp-OutputArea-output,
.jp-RenderedText,
.jp-RenderedHTML,
.output_wrapper,
.output,
.output_area,
div.output_area {
  background: #020617 !important;
  color: #e2e8f0 !important;
}

/* Cell prompts / gutter */
.jp-InputPrompt,
.jp-OutputPrompt,
.input_prompt,
.output_prompt,
.prompt {
  background: #020617 !important;
  color: #334155 !important;
}

/* Notebook toolbar */
.jp-Toolbar,
#maintoolbar,
#maintoolbar-container,
.toolbar {
  background: #0f172a !important;
  border-color: #1e293b !important;
}
.jp-Toolbar-item button,
.btn,
.btn-default {
  background: #1e293b !important;
  color: #94a3b8 !important;
  border-color: #334155 !important;
}

/* Menu bar */
.jp-MenuBar,
#menubar,
#menubar-container {
  background: #020617 !important;
  border-color: #1e293b !important;
}
.jp-Menu,
.p-Menu,
.lm-Menu {
  background: #0f172a !important;
  border-color: #334155 !important;
  color: #e2e8f0 !important;
}

/* Tab bars (file tabs, notebook tabs) */
.jp-DockPanel-tabBar,
.jp-TabBar,
#tabs {
  background: #020617 !important;
  border-color: #1e293b !important;
}

/* Sidebar / file browser */
.jp-SideBar,
.jp-FileBrowser,
.jp-DirListing,
#ipython_notebook {
  background: #0f172a !important;
  color: #94a3b8 !important;
}

/* Scrollbars — global */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #020617; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* ── reset / base ───────────────────────────────────────────────────────── */
.jp-OutputArea, .jp-OutputArea * {
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif !important;
  box-sizing: border-box;
}

/* ── dark widget containers — ALL widget types transparent so parent bg shows */
.widget-vbox, .widget-hbox, .widget-box,
.widget-html, .widget-html-content,
.widget-label, .widget-label-basic,
.widget-readout,
.widget-output,
.jupyter-widgets {
  background: transparent !important;
}

/* ── tab contents — override the jp-layout-color1 variable that makes it white */
.jupyter-widgets.widget-tab > .widget-tab-contents,
.jupyter-widgets.jupyter-widget-tab > .widget-tab-contents {
  background: #0f172a !important;
  border-color: #1e293b !important;
  color: #e2e8f0 !important;
}

/* ── tab bar itself */
.jupyter-widgets.widget-tab > .p-TabBar,
.jupyter-widgets.widget-tab > .lm-TabBar,
.jupyter-widgets.jupyter-widget-tab > .p-TabBar,
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar {
  background: #020617 !important;
  border-color: #1e293b !important;
}
.jupyter-widgets.widget-tab .p-TabBar-tab,
.jupyter-widgets.widget-tab .lm-TabBar-tab,
.jupyter-widgets.jupyter-widget-tab .p-TabBar-tab,
.jupyter-widgets.jupyter-widget-tab .lm-TabBar-tab {
  background: #020617 !important;
  color: #475569 !important;
  border-color: #1e293b !important;
}
.jupyter-widgets.widget-tab .p-TabBar-tab.p-mod-current,
.jupyter-widgets.widget-tab .lm-TabBar-tab.lm-mod-current,
.jupyter-widgets.jupyter-widget-tab .p-TabBar-tab.p-mod-current,
.jupyter-widgets.jupyter-widget-tab .lm-TabBar-tab.lm-mod-current {
  background: #0f172a !important;
  color: #e2e8f0 !important;
  border-bottom-color: #0f172a !important;
}

/* Global text colour for all widget text so it reads on dark bg */
.widget-html *, .widget-html-content *,
.widget-label *, .widget-readout * {
  color: #e2e8f0;
}

/* Specifically keep buttons/inputs from inheriting the colour override */
.widget-button, .widget-text input,
.widget-datepicker input,
.widget-dropdown select,
.widget-select select { color: inherit !important; }

/* ── dark inputs ─────────────────────────────────────────────────────────── */
.widget-text input,
.widget-textarea textarea {
  background: #0f172a !important;
  color: #e2e8f0 !important;
  border: 1px solid #334155 !important;
  border-radius: 6px !important;
  font-size: 12px !important;
  padding: 5px 9px !important;
  transition: border-color 0.15s !important;
}
.widget-text input:focus,
.widget-textarea textarea:focus {
  border-color: #3b82f6 !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.25) !important;
}
.widget-text input::placeholder { color: #475569 !important; }

/* ── dark date picker ────────────────────────────────────────────────────── */
.widget-datepicker input {
  background: #0f172a !important;
  color: #e2e8f0 !important;
  border: 1px solid #334155 !important;
  border-radius: 6px !important;
  font-size: 12px !important;
}

/* ── dark labels ─────────────────────────────────────────────────────────── */
.widget-label, .widget-label-basic {
  color: #94a3b8 !important;
  font-size: 12px !important;
}

/* ── dark select / dropdown ──────────────────────────────────────────────── */
.widget-dropdown select,
.widget-select select,
.widget-selectmultiple select {
  background: #0f172a !important;
  color: #e2e8f0 !important;
  border: 1px solid #334155 !important;
  border-radius: 8px !important;
  font-size: 12px !important;
}
.widget-dropdown select option,
.widget-select select option {
  background: #1e293b !important; color: #e2e8f0 !important;
}
.widget-select select option:checked,
.widget-selectmultiple select option:checked {
  background: #3b82f6 !important; color: #fff !important;
}

/* ── dark toggle buttons ─────────────────────────────────────────────────── */
.widget-toggle-buttons .widget-toggle-button {
  background: #1e293b !important;
  color: #94a3b8 !important;
  border: 1px solid #334155 !important;
  border-radius: 6px !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  transition: all 0.15s !important;
}
.widget-toggle-buttons .widget-toggle-button:hover {
  background: #334155 !important;
  color: #e2e8f0 !important;
}
.widget-toggle-buttons .widget-toggle-button.mod-active {
  background: #3b82f6 !important;
  color: #fff !important;
  border-color: #2563eb !important;
}

/* ── dark toggle button (single) ─────────────────────────────────────────── */
.widget-toggle-button {
  background: #1e293b !important;
  color: #94a3b8 !important;
  border: 1px solid #334155 !important;
}
.widget-toggle-button.mod-active {
  background: #3b82f6 !important;
  color: #fff !important;
}

/* ── dark buttons ────────────────────────────────────────────────────────── */
.widget-button {
  border-radius: 7px !important;
  font-weight: 600 !important;
  font-size: 12px !important;
  letter-spacing: 0.1px !important;
  transition: opacity 0.15s, transform 0.1s !important;
}
.widget-button:not(.mod-info):not(.mod-success):not(.mod-warning):not(.mod-danger) {
  background: #1e293b !important;
  color: #cbd5e1 !important;
  border: 1px solid #334155 !important;
}
.widget-button:not(.mod-info):not(.mod-success):not(.mod-warning):not(.mod-danger):hover {
  background: #334155 !important;
}
.widget-button:active { transform: scale(0.97) !important; }
.widget-button.mod-info    { background: #3b82f6 !important; border-color: #2563eb !important; }
.widget-button.mod-success { background: #22c55e !important; border-color: #16a34a !important; }
.widget-button.mod-warning { background: #f59e0b !important; border-color: #d97706 !important; }
.widget-button.mod-danger  { background: #ef4444 !important; border-color: #dc2626 !important; }

/* ── slider ──────────────────────────────────────────────────────────────── */
.widget-slider .ui-slider .ui-slider-handle {
  background: #3b82f6 !important; border-color: #2563eb !important;
}
.widget-slider .ui-slider { background: #1e293b !important; border-color: #334155 !important; }

/* ── progress ────────────────────────────────────────────────────────────── */
.widget-progress .progress-bar { background: #3b82f6 !important; transition: width 0.3s !important; }
.widget-progress .progress { background: #1e293b !important; }

/* ── tabs ────────────────────────────────────────────────────────────────── */
.p-TabBar-tab, .lm-TabBar-tab {
  font-size: 12px !important;
  font-weight: 600 !important;
  padding: 7px 14px !important;
  color: #64748b !important;
  background: #0f172a !important;
  border-color: #334155 !important;
}
.p-TabBar-tab.p-mod-current, .lm-TabBar-tab.lm-mod-current,
.p-TabBar-tab.p-mod-current, .p-TabBar-tab[aria-selected="true"] {
  color: #e2e8f0 !important;
  background: #1e293b !important;
  border-bottom-color: #1e293b !important;
}
.p-TabBar-content, .lm-TabBar-content {
  background: #0f172a !important;
  border-bottom: 1px solid #334155 !important;
}
.p-TabPanel, .lm-TabPanel { background: #1e293b !important; }

/* ── file upload ─────────────────────────────────────────────────────────── */
.widget-upload button {
  background: #1e293b !important;
  color: #94a3b8 !important;
  border: 1px dashed #334155 !important;
  border-radius: 7px !important;
}

/* ── output widget ───────────────────────────────────────────────────────── */
.widget-output {
  background: #0f172a !important;
}

/* ── app header ─────────────────────────────────────────────────────────── */
.hbv-app-header {
  background: linear-gradient(120deg, #020617 0%, #0f2d55 40%, #1d4ed8 100%);
  color: #fff;
  padding: 12px 20px;
  display: flex;
  align-items: center;
  gap: 14px;
  border-radius: 10px 10px 0 0;
  border-bottom: 1px solid #1e3a5f;
}
.hbv-app-header .icon {
  width: 34px; height: 34px;
  background: rgba(255,255,255,0.12);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 17px; flex-shrink: 0;
}
.hbv-app-header h1 {
  margin: 0; font-size: 15px; font-weight: 700; letter-spacing: -0.3px; color: #f1f5f9;
}
.hbv-app-header .sub { font-size: 10px; opacity: 0.55; margin-top: 1px; color: #cbd5e1; }
.hbv-app-header .badge {
  margin-left: auto; background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 20px; padding: 3px 10px;
  font-size: 11px; font-weight: 700; letter-spacing: 0.5px; white-space: nowrap; color: #93c5fd;
}

/* ── stepper ─────────────────────────────────────────────────────────────── */
.hbv-stepper {
  display: flex; align-items: center;
  padding: 12px 20px;
  background: #0f172a;
  border-bottom: 1px solid #1e293b;
}
.hbv-step-item {
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  min-width: 75px; cursor: default;
}
.hbv-step-conn {
  flex: 1; height: 2px; background: #1e293b; margin-bottom: 20px;
  transition: background 0.3s;
}
.hbv-step-conn.done { background: #16a34a; }
.hbv-step-circle {
  width: 28px; height: 28px; border-radius: 50%;
  background: #1e293b; color: #475569;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 12px;
  border: 2px solid #334155;
  transition: all 0.25s;
}
.hbv-step-item.active .hbv-step-circle {
  background: #3b82f6; color: #fff; border-color: #3b82f6;
  box-shadow: 0 0 0 4px rgba(59,130,246,0.25);
}
.hbv-step-item.done .hbv-step-circle {
  background: #16a34a; color: #fff; border-color: #16a34a;
}
.hbv-step-label {
  font-size: 10px; font-weight: 600; color: #475569;
  text-align: center; white-space: nowrap; letter-spacing: 0.2px;
}
.hbv-step-item.active .hbv-step-label { color: #60a5fa; }
.hbv-step-item.done  .hbv-step-label  { color: #4ade80; }

/* ── section label ───────────────────────────────────────────────────────── */
.hbv-lbl {
  font-size: 10px; font-weight: 700; color: #475569;
  text-transform: uppercase; letter-spacing: 0.9px;
  margin: 12px 0 5px 0;
}
.hbv-lbl:first-child { margin-top: 0; }

/* ── inner card ──────────────────────────────────────────────────────────── */
.hbv-card {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 14px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.35);
}
.hbv-card-flat {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 8px;
  padding: 10px 12px; margin: 6px 0;
}

/* ── chip grid ───────────────────────────────────────────────────────────── */
.hbv-chip-grid {
  display: flex; flex-wrap: wrap; gap: 4px;
  max-height: 150px; overflow-y: auto;
  padding: 8px; margin: 4px 0;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 8px;
  scrollbar-width: thin;
  scrollbar-color: #334155 #0f172a;
}
.hbv-chip-grid::-webkit-scrollbar { width: 6px; }
.hbv-chip-grid::-webkit-scrollbar-track { background: #0f172a; }
.hbv-chip-grid::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

.hbv-chip {
  background: #1e3a5f;
  color: #93c5fd;
  border-radius: 20px; padding: 3px 10px;
  font-size: 11px; font-weight: 600;
  cursor: pointer; user-select: none;
  border: 1.5px solid #1d4ed8;
  transition: all 0.12s; white-space: nowrap;
}
.hbv-chip:hover  { background: #1d4ed8; color: #fff; transform: translateY(-1px); }
.hbv-chip.sel    { background: #c2410c; color: #fff; border-color: #ea580c; }
.hbv-chip-empty  { color: #475569; font-size: 12px; padding: 8px; }

/* ── selected catchment tags ─────────────────────────────────────────────── */
.hbv-sel-tag {
  display: inline-flex; align-items: center; gap: 5px;
  background: #431407; color: #fb923c;
  border: 1.5px solid #c2410c;
  border-radius: 20px; padding: 3px 10px;
  font-size: 11px; font-weight: 700; margin: 2px;
}

/* ── status strip ────────────────────────────────────────────────────────── */
.hbv-status-warn {
  background: #422006; border-left: 3px solid #d97706;
  padding: 8px 12px; border-radius: 0 6px 6px 0; font-size: 12px; color: #fbbf24;
}
.hbv-status-ok {
  background: #052e16; border-left: 3px solid #16a34a;
  padding: 8px 12px; border-radius: 0 6px 6px 0; font-size: 12px; color: #4ade80;
}

/* ── readiness checklist ─────────────────────────────────────────────────── */
.hbv-ready-row {
  display: flex; align-items: center; gap: 8px;
  padding: 5px 0; font-size: 12px;
  border-bottom: 1px solid #1e293b;
  color: #94a3b8;
}
.hbv-ready-row:last-child { border-bottom: none; }

/* ── right panel header ──────────────────────────────────────────────────── */
.hbv-right-header {
  background: #020617;
  color: #94a3b8;
  padding: 10px 16px; font-size: 13px; font-weight: 700;
  display: flex; align-items: center; gap: 8px;
  border-radius: 10px 10px 0 0;
  border-bottom: 1px solid #1e293b;
}
.hbv-right-header .dot {
  width: 8px; height: 8px; background: #22c55e;
  border-radius: 50%; display: inline-block;
  box-shadow: 0 0 6px #22c55e;
}

/* ── divider ─────────────────────────────────────────────────────────────── */
.hbv-hr { border: none; border-top: 1px solid #1e293b; margin: 10px 0; }

/* ── log scrollbar ───────────────────────────────────────────────────────── */
.widget-output::-webkit-scrollbar { width: 6px; }
.widget-output::-webkit-scrollbar-track { background: #0f172a; }
.widget-output::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
"""


def inject_css():
    display(_HTML(_CSS))


# ── Widget helpers ─────────────────────────────────────────────────────────

def app_header_widget():
    return widgets.HTML("""
    <div class="hbv-app-header">
      <div class="icon">🌊</div>
      <div>
        <h1>HBV Hydrological Model</h1>
        <div class="sub">Configure → Download → Run → Analyse</div>
      </div>
      <div class="badge">WE3Unit</div>
    </div>
    """)


def render_stepper(step_idx):
    """Return HTML string for stepper bar. step_idx 0-based."""
    steps = [('🗺', 'Catchment'), ('🌧', 'Climate'), ('🏙', 'Land Use'), ('🚀', 'Run')]
    parts = ['<div class="hbv-stepper">']
    for i, (icon, lbl) in enumerate(steps):
        if i < step_idx:
            cls, circle = 'done', '✓'
        elif i == step_idx:
            cls, circle = 'active', str(i + 1)
        else:
            cls, circle = '', str(i + 1)
        parts.append(
            f'<div class="hbv-step-item {cls}">'
            f'  <div class="hbv-step-circle">{circle}</div>'
            f'  <div class="hbv-step-label">{icon} {lbl}</div>'
            f'</div>'
        )
        if i < 3:
            conn = 'done' if i < step_idx else ''
            parts.append(f'<div class="hbv-step-conn {conn}"></div>')
    parts.append('</div>')
    return ''.join(parts)


def section_lbl(text):
    return widgets.HTML(f'<div class="hbv-lbl">{text}</div>')


def divider():
    return widgets.HTML('<hr class="hbv-hr">')


def tip(text):
    return widgets.HTML(
        f'<div style="font-size:11px;color:#475569;padding:2px 0 4px 0">{text}</div>'
    )


def right_panel_header():
    return widgets.HTML("""
    <div class="hbv-right-header">
      <span class="dot"></span> Activity &amp; Results
    </div>
    """)
