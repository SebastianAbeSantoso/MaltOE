# Kiroween 2026: rules check (pre-flight P3)

First checked 2026-11-16 against the handoff (`docs/elastic-kiroween-handoff.md`, "Kiroween rules and deadlines") from search excerpts, because Devpost returns HTTP 403 to automated fetches. Updated 2026-11-17 from the full text of the Devpost overview page (`kiroween.devpost.com`), pasted by the builder from a browser.

**Not yet seen:** the **Official Rules** tab (eligibility, full judging details) and the **Resources** tab (credit request details). The overview page says the Rules tab has the full details. Nothing here replaces reading it.

## Dates

| Date | Event | Source |
| --- | --- | --- |
| Oct 30, 2026 | Submission period opens | Devpost overview |
| Nov 11, 10:00 AM PT | Devpost Build Session (optional) | Devpost overview |
| Nov 12 to 15 | Epic MegaJam | handoff only, not checked |
| Dec 1, 12:00 PM PT (Dec 2, 04:00 WITA) | Credit requests close | third-party article (P1 is done, so this no longer matters) |
| Dec 4, 2:00 PM PT (Dec 5, 06:00 WITA) | Submission deadline | Devpost overview: "Friday, December 4, 2026 at 2:00 PM Pacific Time" |
| Dec 31 | Redeemed credits expire | third-party article |

PT is UTC-8 in December, so the deadline is 22:00 UTC on Dec 4, which is 06:00 on Dec 5 in WITA (UTC+8). **No difference from the handoff dates.**

## Confirmed from the Devpost overview page

### Eligibility of the project

- Only new projects, built during the Submission Period (Oct 30 to Dec 4, 2026). This repository's first commit is 2026-11-11. **OK.**
- The app must use Kiro: "IDE, CLI, Web, and/or Kiro Crew". **Kiro Web counts.** This settles the earlier open question.
- Teams can have up to three members.

### Category

- One category per submission: Resurrection, Frankenstein, Skeleton Crew, or Costume Contest. MaltOE enters **Resurrection** (Imprecise Computation, 1991, and the Working Set model, 1968).
- A project can win one Overall Prize or one Category Prize, not both, plus any number of Bonus Prizes.

### What to submit

| Item | Rule | MaltOE status |
| --- | --- | --- |
| Public GitHub repo | GitHub only (not GitLab or Bitbucket). Must contain all source, assets and instructions needed to run it. | Public (checked in P2). Instructions: README Quickstart, Task 13.1. |
| Open-source license | A license file that GitHub detects and shows in the repo's About section. | **OK.** GitHub API reports `license: MIT`. |
| `/.kiro` at the repo root | Shows use of specs, hooks and steering. Must not be in `.gitignore` ("could disqualify"). | **OK** (P2). `main` gets the renamed spec path when PRs #2–#4 merge. |
| URL to the working app | Login credentials go in the private submission form. | **Gap.** MaltOE is a local app, not a hosted one. Decide what this URL is (see below). |
| Text description | Features and functionality. | Task 13. |
| Demo video | Under 3 minutes (judges need not watch past 3:00). Show the project **working on the device it was built for**. **Public on YouTube.** No third-party trademarks, copyrighted music or other copyrighted material without permission. | Req 15.3 says under 3 minutes. YouTube, device, and trademark/music rules are **new** (see below). |
| Kiro write-up | How Kiro was used, at a "next-level" depth. Prompts given: IDE/CLI/Web/Crew differences, custom agents, agent hooks, spec-driven development vs vibe coding, steering strategy, MCP, property-based testing, Kiro powers. | **New**, not in Req 15. |

### Judging criteria

Five criteria. The overview page gives no weights.

1. **Potential Impact:** a real problem for a real audience, shown working; useful, intuitive, accessible.
2. **Technological Implementation:** how thoroughly and skilfully the project uses Kiro; strategy, variety of Kiro tools, experimentation.
3. **Design:** a complete, coherent product experience, "not just a technical proof of concept".
4. **Innovation/Idea:** novel concept that differs from existing ones.
5. **Presentation:** the video shows the project working end to end and says what problem it solves, for whom, and why it matters.

### Prizes

1st $30,000, 2nd $20,000, 3rd $10,000 (matches the handoff). Best Resurrection $5,000. Bonus prizes that may apply to MaltOE:

| Bonus prize | Extra step | Applies? |
| --- | --- | --- |
| Best Demo ($1,000) | none, all eligible submissions are considered | yes, automatically |
| Social Blitz (5 winners, 3 months Kiro Pro+) | public social post during the period, starting with `#KiroweenHackathon26`, about your favourite thing about Kiro; add the URL | optional, cheap |
| Bonus Blog (100 winners, 2,000 Kiro credits) | public post on a third-party blog platform during the period, saying it is written for Kiroween 2026; add the URL | optional |
| Best Student Project: Resurrection ($1,500 + 1 year Kiro Pro Max) | all members students; LinkedIn, student email, year, degree, school | only if the builder is a student |
| Best Startup, Ghost of Kiroween Past, Best Kiro Crew App | founder proof / Kiroween 2025 registration / a Kiro Crew app | unlikely |

## Differences from the handoff and from Requirement 15

Dates: **none**.

Rules that Requirement 15 does not cover yet:

1. **License visible in the About section.** Already satisfied (MIT, detected).
2. **URL to a working app.** MaltOE runs locally on the user's GPU and can't be hosted for judges. The likely answer is the repo or a GitHub Release with a Quickstart, plus the video as the proof that it works. If the Official Rules say more about apps that run locally, follow those.
3. **Video on public YouTube, on the device it was built for.** Record on the RTX 4060 laptop. That machine is also where the numbers come from.
4. **No third-party trademarks or copyrighted material in the video without permission.** Use no logos and no copyrighted music. The planned DeepSeek opening joke names another company's product. Keep it to a plain-text mention or drop it; this note is not legal advice.
5. **Kiro write-up.** The judges ask specifically about hooks, specs vs vibe coding, steering, MCP, PBT, powers, custom agents and which Kiro clients were used. MaltOE has specs, steering, two agent hooks, Kiro Web, sub-agents and the github-cli power, and PBT with `hypothesis` is planned (design.md, Correctness Properties). Record notes as the build goes so the write-up isn't reconstructed at the end.
6. **Design criterion: "not just a technical proof of concept".** This supports keeping the web UI (Task 11) in scope and polished, not a debug page.

## Still to confirm (Official Rules and Resources tabs)

- **Eligibility of the entrant:** age, country of residence, employer restrictions. The handoff says not to infer it from the builder's timezone, and this file does not.
- Judging weights and the judging process (the stages, and who judges).
- Anything the Rules tab says about apps that run locally, without a hosted URL.
- Credit details on the Resources tab (moot: P1 is done).
