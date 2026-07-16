"use strict";

const MENU_SELECTION = "audiocpp-read-selection";
const MENU_PAGE = "audiocpp-read-page";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: MENU_SELECTION,
    title: "Read selection",
    contexts: ["selection"],
  });
  chrome.contextMenus.create({
    id: MENU_PAGE,
    title: "Read page",
    contexts: ["page", "selection"],
  });
});

async function pageText(tabId) {
  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId },
    func: () => document.body.innerText.trim(),
  });
  return result || "";
}

async function openPlayer(text, source) {
  const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  await chrome.storage.session.set({
    [id]: {
      text: text.replace(/\s+/g, " ").trim(),
      source,
      autoStart: true,
    },
  });
  await chrome.windows.create({
    url: chrome.runtime.getURL(`player.html?id=${encodeURIComponent(id)}`),
    type: "popup",
    width: 380,
    height: 640,
    focused: true,
  });
}

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (!tab || tab.id === undefined) {
    return;
  }
  (async () => {
    if (info.menuItemId === MENU_SELECTION) {
      await openPlayer(info.selectionText || "", "selection");
    } else if (info.menuItemId === MENU_PAGE) {
      await openPlayer(await pageText(tab.id), "page");
    }
  })().catch((error) => {
    console.error("audio.cpp speaker context menu failed:", error);
  });
});
