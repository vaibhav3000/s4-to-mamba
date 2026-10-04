import { test, expect } from "@playwright/test";

const URL = "/index.html";

function watchErrors(page) {
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  return errors;
}

test("page loads with committed data and charts, no page errors", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto(URL);
  await expect(page).toHaveTitle(/SSM Efficiency & Capability Explorer/);
  await expect(page.locator("#schem-svg .tok")).toHaveCount(8);
  await expect(page.locator("#schem-models button")).toHaveCount(5);
  const data = await page.evaluate(() => ({
    efficiency: window.DATA.efficiency.length,
    parity: Object.keys(window.DATA.parity).length,
  }));
  expect(data.efficiency).toBe(24);
  expect(data.parity).toBeGreaterThanOrEqual(3);
  expect(errors).toEqual([]);
});

test("guided tour starts, steps with keyboard, resets, and exits", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto(URL);
  await page.locator("#tour-open").click();
  const card = page.locator("#tour-card");
  await expect(card).toBeVisible();
  await expect(page.locator("#tour-pos")).toHaveText("step 1 / 6");
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#tour-pos")).toHaveText("step 2 / 6");
  await page.keyboard.press("r");
  await expect(page.locator("#tour-pos")).toHaveText("step 1 / 6");
  await page.keyboard.press("Escape");
  await expect(card).toBeHidden();
  expect(errors).toEqual([]);
});

test("schematic Play/Step/Reset works", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto(URL);
  const pos = page.locator("#schem-pos");
  await page.locator("#schem-step").click();
  await page.locator("#schem-step").click();
  await page.locator("#schem-step").click();
  await page.locator("#schem-step").click();
  await expect(pos).toHaveText("step 4/4");
  await expect(page.locator("#schem-step")).toBeDisabled();
  await page.locator("#schem-prev").click();
  await expect(pos).toHaveText("step 3/4");
  await page.locator("#schem-reset").click();
  await expect(pos).toHaveText("step 0/4");
  const play = page.locator("#schem-play");
  await play.click();
  await expect(play).toHaveAttribute("aria-pressed", "true");
  await play.click();
  await expect(play).toHaveAttribute("aria-pressed", "false");
  expect(errors).toEqual([]);
});

test("efficiency scrubber snaps to a measured length and highlights charts", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto(URL);
  await page.locator("#eff-slider").fill("5");
  await expect(page.locator("#eff-slider")).toHaveAttribute("aria-valuetext", "32768 tokens");
  await expect(page.locator("#eff-read")).toContainText("measured OOM");
  const highlighted = await page.locator('#chart-latency svg line[stroke-dasharray="3 3"]').count();
  expect(highlighted).toBe(1);
  // a measured readout, not an interpolated one
  await expect(page.locator("#eff-read")).toContainText("186");
  expect(errors).toEqual([]);
});

test("task switcher changes the displayed ranking", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto(URL);
  const order = () =>
    page.evaluate(() =>
      [...document.querySelectorAll("#taskchart .taskbar")]
        .sort((a, b) => a.style.order - b.style.order)
        .map((b) => b.dataset.model)
    );
  await page.locator("#task-tabs").getByRole("button", { name: "Exact copy" }).click();
  expect((await order())[0]).toBe("transformer");
  await page.locator("#task-tabs").getByRole("button", { name: "Parity" }).click();
  expect((await order())[0]).toBe("mamba3");
  expect(errors).toEqual([]);
});

test("no horizontal overflow at 390px", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(URL);
  const o = await page.evaluate(() => ({
    sw: document.documentElement.scrollWidth,
    w: document.documentElement.clientWidth,
  }));
  expect(o.sw).toBeLessThanOrEqual(o.w);
});

test("reduced motion: tour opens without autoplay, Step still works", async ({ browser }) => {
  const context = await browser.newContext({ reducedMotion: "reduce" });
  const page = await context.newPage();
  await page.goto(URL);
  await page.locator("#tour-open").click();
  await expect(page.locator("#tour-card")).toBeVisible();
  await expect(page.locator("#tour-play")).toHaveAttribute("aria-pressed", "false");
  await page.locator("#tour-next").click();
  await expect(page.locator("#tour-pos")).toHaveText("step 2 / 6");
  await context.close();
});
