import asyncio
from playwright.async_api import async_playwright

REELS_URL = "https://www.instagram.com/reels/"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state='ig_session.json')
        page = await context.new_page()
        await page.goto(REELS_URL, timeout=60000)
        await page.wait_for_load_state('networkidle')
        print('URL:', page.url)
        print('Title:', await page.title())
        await page.screenshot(path='debug_reels_manual.png', full_page=True)
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
