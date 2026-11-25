# Kiroween 2026: rules check (pre-flight P3)

First checked 2026-11-16 against the handoff (`docs/elastic-kiroween-handoff.md`, "Kiroween rules and deadlines") from search excerpts, because Devpost returns HTTP 403 to automated fetches. Updated 2026-11-17 from the full text of the Devpost overview page and the **Official Rules** (version "Updated November 16, 2026, to include residents of Australia"), both pasted by the builder from a browser. Where the overview page and the Official Rules differ, the Official Rules prevail (Rules §11).

Entrant: **solo, a student** (builder, 2026-11-17).

## Dates (Rules §1)

| Date (Pacific Time) | WITA (UTC+8) | Event |
| --- | --- | --- |
| Oct 13, 2026, 8:30 AM | Oct 13, 23:30 | Registration opens |
| Oct 30, 2026, 9:00 AM | Oct 31, 00:00 | Submission Period opens |
| Dec 1, 2026, 12:00 PM | Dec 2, 04:00 | Credit requests close (P1 done) |
| Dec 4, 2026, 2:00 PM | Dec 5, 06:00 | **Submission and registration close.** Unredeemed credit codes expire. No extensions, no late entries. |
| Dec 16, 2026, 1:00 PM | Dec 17, 05:00 | Judging Period opens |
| Dec 31, 2026, 11:59 PM | Jan 1, 15:59 | Redeemed credits expire |
| Jan 29, 2027, 1:00 PM | Jan 30, 05:00 | Judging Period ends. **The project must stay available to judges, free and unrestricted, until then.** |
| ~Feb 3, 2027, 9:00 AM | ~Feb 4, 01:00 | Winners announced |

October 13 and 30 are in PDT (UTC-7). The December and later dates are in PST (UTC-8).

**No difference from the handoff dates.** The handoff did not give the judging and announcement dates.

## Eligibility (Rules §3)

- Individuals who are at least the **age of majority where they reside**, at the time of entry. Teams of up to 3. A solo entrant is fine.
- **Excluded residents:** where US or local law prohibits taking part or receiving a prize, including Quebec, Argentina, Brazil, Italy, the Philippines, Russia, Cuba, Iran, North Korea, Syria, Belarus, Crimea, the DNR and LNR regions, the UAE, and other countries under comprehensive OFAC sanctions.
- Also excluded: employees, agents and household or immediate family of AWS, Devpost and other Promotional Entities; judges and their employers; and affiliates of any of these.
- Projects built with financial or preferential support from AWS or Devpost are excluded (§4).
- **The builder confirms their own eligibility.** This file does not infer it from the timezone (handoff rule). The Rules have a Bahasa Indonesia translation, if that helps.
- The rules also say Kiro features and models vary by region, and the entrant is responsible for regional restrictions (§4).

## Project requirements (Rules §4)

| Rule | What it means for MaltOE | Status |
| --- | --- | --- |
| A working software application built with Kiro (IDE, CLI, Web, and/or Crew). "Insufficient use of Kiro will result in disqualification." | Built in Kiro Web, with specs, steering, hooks, sub-agents and a power. | in progress |
| One category | **Resurrection** (Imprecise Computation, 1991; Working Set, 1968) | planned |
| **Functionality:** must install and run consistently on its intended platform, and work as shown in the video and text. | The intended platform must be stated: Windows 11, NVIDIA GPU with 8 GB VRAM, CUDA (the RTX 4060 laptop). AMD/Vulkan stays a stretch and is not claimed as supported unless it works. | **new, add to Req 15** |
| **New project only.** Built during the Submission Period. "Must disclose any other pre-existing code or work incorporated." | Disclose: the llama.cpp fork (MIT, pinned commit), Qwen3-30B-A3B (Apache-2.0), eval data (HumanEval, GSM8K, perplexity text), FastAPI, React and other libraries, and the handoff planning document. **Builder: confirm the handoff was written on or after Oct 30, or disclose it as prior planning.** | **new, add to Req 15** |
| **Open source:** allowed if licenses are respected and the project "enhances and builds upon" the underlying product. | MaltOE adds width adaptation and a controller on top of llama.cpp. Keep llama.cpp's MIT notice in `engine/`. | OK if kept |
| **Third-party integrations:** must be authorised under their terms. | Model and dataset licenses must be checked and listed. | **new** |

## Submission requirements (Rules §4)

| Item | Rule | MaltOE status |
| --- | --- | --- |
| URL to the working app | "a link to a website, functioning demo, or a **test build**" (Testing paragraph). | **Plan: a GitHub Release with a Windows test build plus Quickstart.** Judges may judge only from the text, images and video. |
| Text description | Features and functionality. | Task 13 |
| Demo video | Under 3 minutes. Shows the project **functioning on the device it was built for**. **Public on YouTube**, linked in the form. **No third-party trademarks, copyrighted music or other material without permission.** | Req 15.3 covers the length only. The rest is **new.** |
| Public GitHub repo | All source, assets and instructions. An open-source license detected in the About section. | OK: public, MIT detected (P2) |
| `/.kiro` at the repo root | Not in `.gitignore`, or risk disqualification. | OK (P2). `main` gets the renamed spec path when PRs #2–#4 merge. |
| Category and bonus categories | Named in the form. | Resurrection + Best Student Project: Resurrection (+ optional bonuses) |
| Kiro write-up | IDE/CLI/Web/Crew, custom agents, hooks, specs vs vibe coding, steering, MCP, PBT, powers. | **new**, Task 13 |
| Language | English, or English translations of all materials. | OK |
| Ownership | Original work, solely owned, no infringement of IP. Contractors allowed if the work is the entrant's own ideas. | OK |

## Judging (Rules §6)

- **Stage One, pass/fail:** the project "reasonably fits the theme and reasonably applies the required tools". The README and video must make the Resurrection link and the Kiro use obvious early.
- **Stage Two:** five **equally weighted** criteria, 20% (10 points) each: Potential Impact, Technological Implementation (use of Kiro), Design ("not just a technical proof of concept"), Innovation/Idea, Presentation.
- Judging may use expert panels, peer review, **automated AI-driven analysis**, or a mix. The README, write-up and repo should therefore be easy to read cold, without the video.
- Ties are re-scored from scratch at 20 points per criterion.

## Prizes MaltOE can enter (Rules §8)

A project can win **one Overall Prize or one Category Prize, plus any number of Bonus Prizes.**

| Prize | Amount | How it is awarded | MaltOE |
| --- | --- | --- | --- |
| 1st / 2nd / 3rd | $30,000 / $20,000 / $10,000 | highest overall scores | yes |
| Best Resurrection | $5,000 | highest score in the category | yes |
| **Best Student Project: Resurrection** | $1,500 + 1-year Kiro Pro Max | highest-scoring eligible student entry in Resurrection | **yes (solo student).** The form needs LinkedIn, student email, year and degree, and school name. Proof of enrollment may be requested; reply in time or lose eligibility. |
| Best Demo | $1,000 | highest Presentation score among the top 10% overall | automatic |
| Social Blitz | 3-month Kiro Pro+ (5 winners) | post must **start with** `#KiroweenHackathon26`, posted during the period, linked in the form. Judged on "original thoughts vs AI writing" and post assets. The profile must match the submission details. | optional. **Write it yourself.** |
| Bonus Blog | 2,000 Kiro credits (top 100) | public post on a third-party platform during the period, saying it is written for Kiroween 2026 | optional |
| Startup, Ghost of Kiroween Past, Kiro Crew App | | | no |

Prizes need identity verification, a winner affidavit and tax forms (a W-8BEN for non-US residents) returned within 10 business days. Fees and taxes are the winner's.

## Differences from the handoff and Requirement 15

Dates: **none.**

Rules that Requirement 15 does not cover yet (proposed additions, not yet applied to the spec):

1. **Platform statement and functionality:** state the intended platform, and make sure a judge can install and run the test build there.
2. **Testing access until Jan 29, 2027:** keep the repo public and the test build downloadable until then. Tag the submitted state (for example `kiroween-2026-submission`) so later commits don't change what judges see.
3. **Disclosure of pre-existing code and work:** llama.cpp, the model, datasets, libraries and the handoff document, with licenses.
4. **Video:** public on YouTube, filmed on the RTX 4060 laptop, no copyrighted music, and no third-party trademarks without permission.
5. **Kiro write-up**, covering the listed features.
6. **Bonus entries:** Best Student Project: Resurrection details in the form.

## Open question: trademarks in the video

The rule says the video "must not include third party trademarks ... unless the Entrant has permission". A local-inference demo naturally shows names such as Qwen, NVIDIA, GeForce, Windows and llama.cpp, and the planned opening joke names DeepSeek. The rule does not say whether plain naming of a product (as opposed to logos or branding) counts.

The Rules (§11) say an entrant who thinks a term is ambiguous must send a written request for clarification. **Suggested action:** email `support@devpost.com` before recording. Until there is an answer: show no logos, keep product names to plain text where needed, and drop the DeepSeek joke. This is not legal advice.
