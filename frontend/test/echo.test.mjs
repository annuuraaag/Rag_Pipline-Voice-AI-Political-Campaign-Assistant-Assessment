// Unit tests for telling the user's voice from the answer's echo:  npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { dropWords, interruption, isStopCommand } from "../src/lib/voice/echo.js";

const ANSWER = "The Pratibha Scholarship gives students from families earning under 3 lakh rupees a year "
  + "a grant of 25,000 rupees for higher education. Students must score at least 75 percent in Class 12.";

test("the answer's own voice is not an interruption", () => {
  assert.equal(interruption("the pratibha scholarship gives students from families earning", ANSWER), null);
  assert.equal(interruption("students must score at least 75 percent", ANSWER), null);
  assert.equal(interruption("", ANSWER), null);
});

test("one or two misheard echo words are not an interruption", () => {
  assert.equal(interruption("the pratibha scholarship gives students from families burning", ANSWER), null);
  assert.equal(interruption("a grant of 25,000 rupees for hire education", ANSWER), null);
});

test("the user talking after echo is an interruption, whatever came before", () => {
  // Long echo first: comparing the whole transcript would call this echo (most of it is).
  const heard = "the pratibha scholarship gives students from families earning under 3 lakh rupees a year what about farmers in guntur";
  const cut = interruption(heard, ANSWER);
  assert.ok(cut);
  assert.equal(cut.text, "what about farmers in guntur");
  assert.equal(dropWords(heard, cut.skip), "what about farmers in guntur");
});

test("with headphones (no echo) a short question interrupts", () => {
  assert.deepEqual(interruption("what about healthcare", ANSWER), { skip: 0, text: "what about healthcare" });
  assert.equal(interruption("what", ANSWER), null);
});

test("stop words interrupt at once, and are not a question", () => {
  assert.ok(interruption("the pratibha scholarship gives stop", ANSWER));
  assert.ok(isStopCommand("Stop."));
  assert.ok(isStopCommand("okay stop please"));
  assert.ok(isStopCommand("wait"));
  assert.ok(!isStopCommand("wait what about education"));
  assert.ok(!isStopCommand("what is the plan"));
});

test("dropWords keeps the rest of the text as written", () => {
  assert.equal(dropWords("Rs. 25,000 grant — who is eligible?", 3), "grant — who is eligible?");
  assert.equal(dropWords("who is eligible", 0), "who is eligible");
});
