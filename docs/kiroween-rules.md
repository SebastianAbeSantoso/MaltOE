# Kiroween 2026: rules check (pre-flight P3)

Checked 2026-11-16 against the handoff (`docs/elastic-kiroween-handoff.md`, "Kiroween rules and deadlines").

**Limit of this check:** the Devpost pages (`kiroween.devpost.com` and its `/rules` page) return HTTP 403 to automated fetches, so I could not read the rules text. I read search-result excerpts of the official Devpost page and Kiro's events page, and one third-party article for the credit process. **Please open the Devpost page and its Rules tab in a browser and check the items marked "confirm" below.** Nothing here replaces reading the official rules.

## Confirmed from official-page excerpts

| Item | Handoff | Official page excerpt | Result |
| --- | --- | --- | --- |
| Submission deadline | Dec 4, 2026, 2:00 PM PT | "Friday, December 4, 2026 at 2:00 PM Pacific Time" ([Devpost](https://kiroween.devpost.com/)); "submit by December 4 at 2:00 PM PT" ([Kiro events](https://kiro.dev/events/)) | **matches** |
| Same moment in WITA | Dec 5, 2026, 06:00 | PT is UTC-8 in December, so 22:00 UTC on Dec 4, which is 06:00 on Dec 5 in WITA (UTC+8) | **matches** |
| Only new projects | yes | "Only new projects are eligible. You must build your project during the Submission Period (October 30 to December 4, 2026). Projects you started before October 30, including Kiroween 2025 projects, are not eligible." | **matches**. This repository's first commit is 2026-11-11, inside the period. |
| Prize pool | $100,000 | "$100,000 in prizes" / "$100,000 in cash prizes" | **matches** (the $30,000 / $20,000 / $10,000 split was not in the excerpts: confirm) |
| Categories | Resurrection, Frankenstein, Skeleton Crew, Costume Contest | All four appear in excerpts and in other entrants' write-ups | **matches** |
| Kiro Crew | not mentioned | The Devpost page lists working "across a crew of agents in Kiro Crew" among the Kiro features it invites entrants to use, next to specs, hooks, steering, powers, and MCP | **new**: using Kiro Crew is explicitly welcome |

## From a third-party article (not official)

Source: [revesery.com, "How to Get 4,000 Free Kiro AI Coding Credits"](https://revesery.com/@gican/how-to-get-4000-free-kiro-ai-coding-credits-mvzv4tjq). Treat as unverified until you see it on Devpost.

| Item | Handoff | Article | Result |
| --- | --- | --- | --- |
| Credit request deadline | Dec 1, 12:00 PM PT (Dec 2, 04:00 WITA) | "requests close December 1 at 12 PM PT" | **matches** |
| Amount | 4,000 credits | 4,000 credits, about $160 | matches |
| How | not stated | Register for the hackathon first; use the credit request form under the **Resources** tab on the Devpost page; you need your Kiro user ID (or AWS account ID) | **new** |
| Delay | not stated | A redemption code "generally arrives within 48 business hours" | **new**: do not wait |
| Redeem by | not stated | Redeem the code by December 4; credits expire December 31, 2026 | **new** |
| Limits | not stated | One grant per eligible participant; regional and age restrictions apply | **new**: confirm you are eligible |

## Not in any excerpt (confirm on Devpost)

- Requirement for a **public GitHub repository** with the `.kiro` folder included.
- Requirement for a **demo video under 3 minutes**, and where it must be hosted.
- The exact **judging criteria** and the weight of each.
- **Eligibility** (age, country, team size, employer restrictions). The handoff says not to infer eligibility from the builder's timezone, and I have not.
- Whether the **category** is chosen at submission and whether one project may enter more than one.
- Whether work done in **Kiro Web** counts as "built with Kiro" the same as the IDE. I would expect so, but the excerpts do not say.

## Differences from the handoff

None found in what could be read. Three additions worth acting on:

1. The credit request needs prior Devpost registration and takes up to about two business days, so request it **now**, not on December 1. This is pre-flight P1 and only you can do it.
2. Kiro Crew is named on the official page, which settles the earlier question about whether using it is acceptable.
3. The credit amount you receive must be redeemed before December 4 to be useful for the build, and it expires on December 31.

## Dates (all times as stated above)

| Date | Event |
| --- | --- |
| Oct 30, 2026 | Submission period opens |
| Nov 12 to 15 | Epic MegaJam (from the handoff; not checked) |
| Dec 1, 12:00 PM PT (Dec 2, 04:00 WITA) | Credit requests close (third-party source) |
| Dec 4, 2:00 PM PT (Dec 5, 06:00 WITA) | Submission deadline |
| Dec 31 | Redeemed credits expire (third-party source) |
