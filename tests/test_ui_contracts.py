"""Source-level regression guards, not a substitute for a browser layout pass."""
import ast
from html.parser import HTMLParser
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET
from app.identity import APP_NAME, DESKTOP_ID, ICON_FILE

ROOT = Path(__file__).resolve().parents[1]


class IDs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs):
        value = dict(attrs).get("id")
        if value:
            self.ids.append(value)


def declarations(selector):
    result = {}
    css = (ROOT / "app/web/style.css").read_text()
    css = re.sub(r'/\*.*?\*/','',css,flags=re.S)
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        if selector in [s.strip() for s in match[1].split(",")]:
            for entry in match[2].split(";"):
                if ":" in entry:
                    key, value = entry.split(":", 1)
                    result[key.strip()] = value.strip()
    return result


class UIContractTests(unittest.TestCase):
    def test_visual_pytorch_brand_preserves_compatibility_ids_and_vector_style(self):
        self.assertEqual(APP_NAME, 'Visual PyTorch')
        self.assertEqual(DESKTOP_ID, 'tensor-loom')
        html=(ROOT/'app/web/index.html').read_text()
        js=(ROOT/'app/web/editor.js').read_text()
        self.assertNotIn('Tensor Loom',html)
        self.assertNotIn('Tensor Loom',js)
        self.assertIn('<span>Visual PyTorch<small>',html)
        self.assertIn(' · Visual PyTorch',js)
        svg=ET.parse(ROOT/ICON_FILE).getroot()
        ns={'svg':'http://www.w3.org/2000/svg'}
        self.assertIn('Visual PyTorch', svg.find('svg:title',ns).text)
        self.assertEqual(svg.attrib['viewBox'],'0 0 96 96')
        self.assertEqual(svg.find("svg:path[@id='v-path']",ns).attrib['d'],'M24 22L48 44L72 22')
        self.assertEqual(svg.find("svg:path[@id='t-path']",ns).attrib['d'],'M24 62H72M48 62V80')
        self.assertEqual(len(svg.findall('svg:g/svg:rect',ns)),6)
        self.assertEqual(svg.find('svg:rect',ns).attrib['rx'],'22')

    def test_node_resize_uses_saved_geometry_and_bias_precedes_truncated_stats(self):
        js=(ROOT/'app/web/editor.js').read_text()
        self.assertIn('width:G.width(n)',js)
        self.assertIn('n.x + G.width(n)',js)
        self.assertIn("class:'node-resize'",js)
        self.assertIn("'Bias '+(n.bias!==false?'on':'off')",js)
        self.assertIn('shortened(note,4.9,14)',js)
        self.assertEqual(declarations('.node-resize')['cursor'],'nwse-resize')

    def test_all_palettes_start_closed_under_one_pack(self):
        html=(ROOT/"app/web/index.html").read_text()
        parser=IDs();parser.feed(html)
        self.assertEqual(len(parser.ids),len(set(parser.ids)))
        for target in ("architectures-panel","library-panel","inspector-panel","experiment-dock"):
            self.assertRegex(html,r'<(?:aside|div) id="'+target+r'"[^>]* hidden>')
            self.assertIn('data-open-panel="'+target+'"',html)
        self.assertIn('id="pack-toggle"',html)
        self.assertEqual(declarations(".floating-panel")["position"],"absolute")
        self.assertEqual(declarations(".floating-panel")["border-radius"],"11px")

    def test_launcher_starts_flush_top_left_with_doubled_outer_spacing(self):
        self.assertEqual(declarations('.menu-pack')['left'], '0')
        self.assertEqual(declarations('.menu-pack')['top'], '0')
        self.assertEqual(declarations('#pack-menus')['gap'], 'var(--menu-unit)')
        self.assertEqual(declarations('#pack-menus')['padding'],
                         'var(--menu-unit) calc(2 * var(--menu-unit)) calc(2 * var(--menu-unit))')
        self.assertEqual(declarations('#pack-menus button')['padding'], 'calc(.75 * var(--menu-unit))')

    def test_math_serif_stack_and_metallic_surfaces_are_theme_scoped(self):
        css=(ROOT/'app/web/themes.css').read_text()
        self.assertRegex(css, r':root\[data-theme=math\] body\{font-family:[^}]*"Noto Serif"[^}]*,serif\}')
        self.assertIn(':root[data-theme=metallic]', css)
        self.assertIn('linear-gradient(135deg,var(--panel),var(--chrome))', css)
        self.assertIn('ui-monospace,SFMono-Regular,Consolas,monospace', (ROOT/'app/web/style.css').read_text())

    def test_inspector_cannot_delete_architectures_and_block_command_uses_clicked_id(self):
        html=(ROOT/'app/web/index.html').read_text()
        js=(ROOT/'app/web/editor.js').read_text()
        self.assertNotIn('delete-model', html)
        self.assertNotIn('delete-model', js)
        self.assertIn('contextAt(x,y)', js)
        self.assertIn('canDelete:!busy() && !!node(selected) && doc.graph.nodes.length>2', js)
        self.assertIn('if(doc.id===modelId && node(blockId))removeNode(blockId)', js)

    def test_size_fields_do_not_enforce_run_limits_on_editing(self):
        html=(ROOT/"app/web/index.html").read_text()
        js=(ROOT/"app/web/editor.js").read_text()
        for source in (html,js):
            self.assertNotIn('max="64"',source)
            self.assertNotIn('max="5000"',source)
            self.assertNotIn('max="10000"',source)
        self.assertIn('id="constrained-mode"',html)
        self.assertIn('safe("setConstrainedMode",{enabled})',js)
        self.assertIn("description?.run_safety?.allowed === false",js)

    def test_scrollbars_overlay_content_instead_of_reserving_panel_space(self):
        self.assertEqual(declarations(".overlay-scrollbar")["position"],"absolute")
        self.assertEqual(declarations(".overlay-scrollbar")["right"],"4px")
        self.assertEqual(declarations(".floating-panel>.scrollport")["scrollbar-width"],"none")
        js=(ROOT/"app/web/panels.js").read_text()
        self.assertIn('thumb.setAttribute("role","scrollbar")',js)
        self.assertIn("PageDown:",js)

    def test_architecture_toolbar_is_in_topbar_not_above_canvas(self):
        html=(ROOT/"app/web/index.html").read_text()
        self.assertEqual(html.count('class="canvas-toolbar"'),1)
        self.assertLess(html.index('class="canvas-toolbar"'),html.index("</header>"))
        self.assertLess(html.index('id="model-title"'),html.index('<main id="editor"'))

    def test_architecture_list_has_no_nested_scroll_or_redundant_collapse(self):
        rules = declarations(".model-list")
        self.assertEqual(rules["flex"], "0 0 auto")
        self.assertEqual(rules["max-height"], "none")
        self.assertEqual(rules["overflow"], "visible")
        self.assertEqual(declarations(".library")["overflow"], "hidden")
        self.assertEqual(declarations(".floating-panel>.scrollport")["overflow"], "auto")
        html=(ROOT/'app/web/index.html').read_text()
        self.assertNotIn('id="toggle-architectures"',html)
        self.assertNotIn('Workspace models',html)
        header=html.split('id="architectures-panel"')[1].split('id="architectures-scroll"')[0]
        self.assertIn('id="add-model"',header)
        self.assertIn('id="architecture-bookkeeping"',html)
        js=(ROOT/'app/web/editor.js').read_text()
        self.assertIn('renderBookkeeping();',js)
        self.assertIn("name.textContent=doc.name",js)

    def test_box_icons_are_top_right_and_have_explicit_fold_gestures(self):
        js=(ROOT/'app/web/editor.js').read_text()
        self.assertIn("m.width-m.padding-20",js)
        self.assertIn('window.LoomTypeIcons.svg(n.type)',js)
        self.assertEqual(declarations('.node-type-icon')['pointer-events'],'auto')
        self.assertIn("function foldBlock(nid,collapsed)",js)

    def test_menu_content_is_compact_without_shrinking_readable_fonts(self):
        rules=declarations('.floating-panel .palette')
        self.assertEqual(rules['padding'],'var(--menu-unit) calc(1.25 * var(--menu-unit))')
        self.assertEqual(rules['gap'],'var(--menu-unit)')
        self.assertEqual(declarations('.palette b')['font-size'],'calc(var(--ui-font-size,14px) - 1px)')
        self.assertEqual(declarations(':root')['--menu-unit'],'4px')
        js=(ROOT/'app/web/editor.js').read_text()
        self.assertIn("style.setProperty('--menu-unit',(state.display?.spacing ?? 4)+'px')",js)
        self.assertNotIn('data-spacing=compact',(ROOT/'app/web/style.css').read_text())

    def test_menu_spacing_never_sets_diagram_text_or_geometry(self):
        css=(ROOT/'app/web/style.css').read_text()
        for selector,body in re.findall(r'([^{}]+)\{([^{}]*)\}',css):
            if '--menu-unit' in body and selector.strip()!=':root':
                self.assertNotIn('.node',selector)
                self.assertNotIn('font-size',body)
                for entry in body.split(';'):
                    if '--menu-unit' in entry:
                        self.assertNotIn('font-size',entry)
                        self.assertNotIn('line-height',entry)
        js=(ROOT/'app/web/graph-tools.js').read_text()
        self.assertNotIn('menu-unit',js)

    def test_mathjax_is_offline_opt_in_and_notifications_have_dismiss_controls(self):
        html=(ROOT/'app/web/index.html').read_text()
        self.assertIn('src="vendor/mathjax/MathJax.js?config=TeX-AMS_SVG-full"',html)
        self.assertIn('id="toast-close"',html)
        self.assertIn('aria-label="Dismiss notification"',html)
        self.assertRegex((ROOT/'app/web/style.css').read_text(),r'#toast\{[^{}]*transition:opacity \.25s ease')
        base=ROOT/'app/web/vendor/mathjax'
        for file in ('MathJax.js','LICENSE','DEBIAN-COPYRIGHT','config/TeX-AMS_SVG-full.js',
                     'extensions/Safe.js','jax/output/SVG/fonts/TeX/fontdata.js'):
            self.assertTrue((base/file).is_file(),file)

    def test_validation_banners_anchor_to_the_right(self):
        self.assertEqual(declarations('.validation-chip')['left'],'auto')
        self.assertEqual(declarations('.validation-chip')['right'],'16px')

    def test_creation_preset_is_absent_and_architectures_are_separate(self):
        parser = IDs()
        parser.feed((ROOT / "app/web/index.html").read_text())
        self.assertNotIn("new-mlp",parser.ids)
        self.assertNotIn("mlp-dialog",parser.ids)
        html=(ROOT/'app/web/index.html').read_text()
        library=html.split('id="library-panel"')[1].split('</aside>')[0]
        self.assertNotIn('id="model-list"',library)
        self.assertIn('id="reference-select"',library)
        self.assertEqual(declarations(".model-list")["margin-right"],"0")

    def test_dialogs_use_explicit_titles_not_browser_origin_titles(self):
        html=(ROOT/"app/web/index.html").read_text()
        js=(ROOT/"app/web/editor.js").read_text()
        self.assertIn('aria-labelledby="confirm-title"',html)
        self.assertIn('confirmDialog("Disable constrained mode?"',js)
        self.assertNotRegex(js,r"(?<![A-Za-z])confirm\(")

    def test_launcher_uses_vector_icon_and_content_size_at_large_fonts(self):
        html=(ROOT/'app/web/index.html').read_text()
        pack=html.split('id="pack-toggle"')[1].split('</button>')[0]
        self.assertIn('<svg',pack)
        self.assertIn('aria-hidden="true"',pack)
        self.assertNotIn('▦',pack)
        self.assertEqual(declarations('.menu-pack:not(.collapsed)')['width'],'max-content')
        self.assertEqual(declarations('#pack-menus button')['white-space'],'nowrap')
        js=(ROOT/'app/web/panels.js').read_text()
        self.assertIn('new ResizeObserver(onResize).observe(pack)',js)

    def test_ram_policy_is_shown_without_old_feature_caps(self):
        js=(ROOT/'app/web/editor.js').read_text()
        self.assertIn('memory.estimated_bytes',js)
        self.assertIn('memory.budget_bytes',js)
        self.assertNotIn('limits.input_features',js)

    def test_compact_blocks_keep_descriptions_as_accessible_tooltips(self):
        self.assertEqual(declarations('.palette small')['display'],'none')
        self.assertEqual(declarations('.palette>span:nth-child(3)')['display'],'none')
        js=(ROOT/'app/web/editor.js').read_text()
        self.assertIn('button.title=description',js)
        self.assertIn("button.setAttribute('aria-label',name+'. '+description)",js)

    def test_right_click_selects_without_starting_a_drag(self):
        js=(ROOT/'app/web/editor.js').read_text()
        right_click=js.split('if(event.button===2){',1)[1].split('if (!doc || busy()',1)[0]
        self.assertIn("if(doc && wire)selectWire(wireFromElement(wire));else selectContextBox(event.target);",right_click)
        self.assertIn('return;',right_click)
        self.assertNotIn('setPointerCapture',right_click)
        self.assertNotIn('drag=',right_click)
        self.assertIn('selectAt(x,y){return selectContextBox(document.elementFromPoint(x,y));}',js)

    def test_renderer_policy_precedes_application_and_window_creation(self):
        tree = ast.parse((ROOT / "app/desktop.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = [n.value.func.id for n in main.body if isinstance(n, ast.Assign) and
                 isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)]
        self.assertLess(calls.index("configure_renderer"), calls.index("QApplication"))
        self.assertLess(calls.index("QApplication"), calls.index("Window"))
