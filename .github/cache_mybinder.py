import asyncio
import playwright
from playwright.async_api import async_playwright

TIMEOUT = 10 * 60 * 1000  # 8 minutes


async def handle_route(route):
    print("Replacing bundle.js")
    response = await route.fetch()
    body = await response.text()
    body = body.replace("var i=[];e.o", "var i=window.myglobal=[];e.o")
    await route.fulfill(
        response=response,
        body=body,
        headers=response.headers,
    )


async def prefetch_binder(
    url="https://mybinder.org/v2/gh/microsoft/sammo/main?urlpath=tree/docs/tutorials/quickstart.ipynb",
):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        await page.route("**/bundle.js*", handle_route)
        await page.goto(url)

        old_log = []
        while True:
            try:
                current_log = await page.evaluate("() => window.myglobal || []")
                current_log = [x for x in current_log if x.strip() != ""]
                if current_log != old_log:
                    print("".join(current_log[len(old_log) :]), flush=True)
                    old_log = current_log
                await asyncio.sleep(1)
            except playwright._impl._errors.Error:
                print(f"Redirected to {page.url}")
                break
        await browser.close()


asyncio.get_event_loop().run_until_complete(prefetch_binder())
