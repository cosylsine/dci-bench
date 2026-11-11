"""Run with Playwright available on PYTHONPATH; no model or external network calls."""
from pathlib import Path
import argparse
import hashlib
import html
import json

from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, default=root / 'docs/project-progress-20261028')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
report = args.report.resolve()
artifact = args.output.resolve()
artifact.mkdir(parents=True, exist_ok=False)
source = (report / 'index.html').read_text()
coverage = json.loads((report / 'benchmark-v2-coverage.json').read_text())
ready = coverage['ready_tasks']
blocked = coverage['blocked_tasks']
zh_ready = sum(row['status'] == 'ready' and row['language'] == 'zh' for row in coverage['tasks'])
blocked_task = next((row['task'] for row in coverage['tasks'] if row['status'] == 'blocked'), None)
checks = []
errors = []


def check(name, condition):
    checks.append({'name': name, 'passed': bool(condition)})
    assert condition, name


def navigation(page_or_frame, name):
    for target in ['workflow', 'data', 'results', 'evidence', 'benchmark-v2']:
        old_url = page_or_frame.url
        page_or_frame.locator(f'.navlinks a[href="#{target}"]').click()
        page_or_frame.wait_for_function("""id => {
            const node = document.getElementById(id);
            return document.activeElement === node &&
              node.getBoundingClientRect().top >= 0 &&
              node.getBoundingClientRect().top < innerHeight;
        }""", arg=target)
        check(f'{name}_{target}', page_or_frame.url == old_url)
    page_or_frame.locator('.brand span').click()
    page_or_frame.wait_for_function('scrollY < 2')
    check(f'{name}_top', page_or_frame.evaluate('document.activeElement.id') == 'top')


with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, args=['--no-sandbox'])
    context = browser.new_context(viewport={'width': 1440, 'height': 1000}, reduced_motion='reduce')
    page = context.new_page()
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto((report / 'index.html').as_uri())
    check('default_10_domains_20_cells', page.locator('#matrix tbody tr').count() == 10 and page.locator('#matrix button').count() == 20)
    check('prepared_and_blocked_match_audit', page.locator('#matrix button.ready').count() == ready and page.locator('#matrix button.blocked').count() == blocked)
    check('languages_only_en_zh', page.locator('#matrix thead th').all_text_contents() == ['领域 / 语言', '英文 en', '中文 zh'])
    check('prepared_query_totals', f"开发{coverage['prepared']['dev']} / 正式{coverage['prepared']['test']}题" in page.locator('#filter-count').inner_text())
    check('no_old_inventory_controls', page.locator('#stage').inner_text().find('三档') == -1 and page.locator('#domain option').count() == 11)
    check('unique_ids', page.evaluate('new Set([...document.querySelectorAll("[id]")].map(n=>n.id)).size === document.querySelectorAll("[id]").length'))
    navigation(page, 'standalone')
    page.emulate_media(reduced_motion='no-preference')
    page.locator('.navlinks a[href="#data"]').click()
    page.wait_for_function('Math.abs(document.getElementById("data").getBoundingClientRect().top - 115) < 2')
    check('smooth_navigation', page.evaluate('document.activeElement.id') == 'data')
    page.emulate_media(reduced_motion='reduce')
    page.locator('a.primary-link').click()
    page.locator('#stage').select_option('blocked')
    check('blocked_filter_keeps_coordinates', page.locator('#matrix tbody tr').count() == 10 and page.locator('#matrix button').count() == blocked)
    if blocked_task is not None:
        page.locator(f'#matrix button[data-task="{blocked_task}"]').click()
        check('blocked_drilldown', page.locator('#sources tbody tr').count() == 1 and '0 / 0 / 0' in page.locator('#sources tbody').inner_text())
    else:
        check('blocked_filter_empty_state', page.locator('#matrix tbody td').count() == 20 and '没有匹配任务' in page.locator('#sources tbody').inner_text())
    page.locator('#reset').click()
    check('reset_20_tasks', page.locator('#sources tbody tr').count() == 20)
    page.locator('#stage').select_option('ready')
    check('all_ready_filter', page.locator('#sources tbody tr').count() == ready)
    page.locator('#reset').click()
    page.locator('#language').select_option('zh')
    check('zh_filter', page.locator('#sources tbody tr').count() == 10)
    page.locator('#stage').select_option('ready')
    check('zh_ready_filter', page.locator('#sources tbody tr').count() == zh_ready)
    page.locator('#reset').click()
    page.locator('#search').fill('不存在的来源')
    check('empty_filter', page.locator('#sources tbody tr.empty-state').count() == 0 and '没有匹配任务' in page.locator('#sources tbody').inner_text() and page.locator('#matrix tbody td').count() == 20)
    page.locator('#search').fill('科研')
    check('chinese_domain_search', page.locator('#sources tbody tr').count() == 2)
    page.locator('#reset').click()
    page.locator('#matrix-view').click()
    page.locator('#matrix button[data-task="finance_en"]').click()
    check('ready_drilldown', page.locator('#sources tbody tr').count() == 1 and '10 / 50 / 1' in page.locator('#sources tbody').inner_text())
    page.locator('#reset').click()
    page.locator('#matrix-view').click()
    if any(row['task'] == 'web_en' and row['status'] == 'ready' for row in coverage['tasks']):
        page.locator('#matrix button[data-task="web_en"]').click()
        check('bcp_ready_drilldown', '10 / 50 / 1' in page.locator('#sources tbody').inner_text() and 'gapfill_browsecomp_plus_en' in page.locator('#sources tbody').inner_text())
        page.locator('#reset').click()
        page.locator('#matrix-view').click()
    page.locator('.navlinks a[href="#data"]').click()
    check('desktop_no_page_overflow', page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
    page.screenshot(path=str(artifact / 'desktop-coverage.png'))
    page.locator('.navlinks a[href="#workflow"]').focus()
    page.keyboard.press('Enter')
    page.wait_for_function('document.activeElement.id === "workflow"')
    check('keyboard_navigation', page.evaluate('document.activeElement.id') == 'workflow')
    page.locator('#workflow summary').click()
    check('budget_details', page.locator('#workflow details').get_attribute('open') is not None)
    page.locator('#workflow summary').click()
    page.set_viewport_size({'width': 390, 'height': 844})
    navigation(page, 'mobile')
    page.locator('.navlinks a[href="#data"]').click()
    check('mobile_no_page_overflow', page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
    page.screenshot(path=str(artifact / 'mobile-coverage.png'))

    # Simulate an embedded HTML preview: srcdoc, opaque sandbox origin, inherited
    # or injected host base URL. Route any attempted navigation locally.
    embedded = context.new_page()
    embedded.on('pageerror', lambda error: errors.append(str(error)))
    embedded.route('**/*', lambda route: route.fulfill(status=200, content_type='text/html', body='<p id="host-route">Host document, not report</p>'))
    host_url = 'https://report-preview.invalid/document'
    def embed(markup):
        markup = markup.replace('<head>', '<head><base href="' + host_url + '">', 1)
        embedded.set_content('<iframe id="preview" sandbox="allow-scripts" style="width:100%;height:850px;border:0" srcdoc="' + html.escape(markup, quote=True) + '"></iframe>')
        frame = embedded.frame_locator('#preview')
        frame.locator('.navlinks').wait_for()
        return embedded.frames[1]
    old = embed((report / 'history/20261028/index.html').read_text())
    old.locator('.navlinks a[href="#data"]').click()
    old.wait_for_url(host_url + '#data')
    check('reproduced_native_fragment_failure_with_host_base', old.locator('#host-route').count() == 1)
    frame = embed(source)
    navigation(frame, 'sandbox_srcdoc_host_base')
    frame.locator('.navlinks a[href="#data"]').click()
    frame.locator('#stage').select_option('blocked')
    check('embedded_filters', frame.locator('#matrix button').count() == blocked)
    check('embedded_document_preserved', frame.url == 'about:srcdoc')
    embedded.screenshot(path=str(artifact / 'embedded-coverage.png'))

    nojs = browser.new_context(java_script_enabled=False, viewport={'width': 390, 'height': 844})
    fallback = nojs.new_page()
    fallback.goto((report / 'index.html').as_uri())
    check('no_script_all_tasks_readable', fallback.locator('#sources tbody tr').count() == 20 and fallback.locator('#table-panel').is_visible())
    check('no_script_dead_controls_hidden', not fallback.locator('#coverage-controls').is_visible())
    check('no_script_no_page_overflow', fallback.evaluate('document.documentElement.scrollWidth <= innerWidth'))
    check('no_javascript_errors', not errors)
    result = {'passed': all(c['passed'] for c in checks), 'checks': checks, 'errors': errors,
              'browser': browser.version, 'html_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'scope': 'Standalone desktop/mobile, sandboxed srcdoc with a host base URL, no-script fallback.',
              'yuque_host_verified': False, 'historical_html_sha256': hashlib.sha256((report / 'history/20261028/index.html').read_bytes()).hexdigest()}
    (artifact / 'browser-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'checks': len(checks), 'errors': errors}, ensure_ascii=False))
    browser.close()
