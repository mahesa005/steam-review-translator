# Task 2 tables for the paper and slides

## Dataset statistics

| split | games | reviews | sentences | no aspect | labels/sent | labels | words/sent |
|---|---|---|---|---|---|---|---|
| train | 9 | 675 | 5150 | 15.0% | 1.05 | 5430 | 14.5 |
| val | 2 | 150 | 1001 | 12.0% | 1.11 | 1108 | 15.4 |
| test | 2 | 150 | 851 | 14.7% | 1.08 | 917 | 13.8 |
| all | 13 | 975 | 7002 | 14.5% | 1.06 | 7455 | 14.6 |

Games per split:

- train: Alien Rage - Unlimited, Battlefleet Gothic: Armada (Classic), Company of Heroes 2, HITMAN™, LARA CROFT AND THE TEMPLE OF OSIRIS™, TRANSFORMERS: Rise of the Dark Spark, Test Drive Unlimited 2, Tony Hawks Pro Skater HD, Toren
- val: Might & Magic: Heroes VI, Paper Sorcerer
- test: Sniper: Ghost Warrior 2, Two Worlds II HD

## Label distribution (sentences with the aspect; share of positive/negative/neutral)

| aspect | train | val | test | total | pos | neg | neu |
|---|---|---|---|---|---|---|---|
| gameplay | 1302 | 268 | 232 | 1802 | 31% | 51% | 18% |
| overall | 1133 | 244 | 243 | 1620 | 57% | 36% | 7% |
| performance | 556 | 191 | 76 | 823 | 9% | 84% | 6% |
| graphics | 410 | 66 | 58 | 534 | 58% | 30% | 11% |
| content | 423 | 43 | 50 | 516 | 33% | 54% | 13% |
| story | 392 | 51 | 62 | 505 | 34% | 42% | 24% |
| other | 277 | 88 | 66 | 431 | 16% | 75% | 8% |
| price | 253 | 54 | 44 | 351 | 26% | 59% | 15% |
| multiplayer | 307 | 4 | 33 | 344 | 24% | 65% | 12% |
| difficulty | 226 | 52 | 32 | 310 | 25% | 42% | 34% |
| audio | 151 | 47 | 21 | 219 | 53% | 40% | 8% |

## Training data examples (train split)

| game | sentence | labels |
|---|---|---|
| Toren | There are a few downsides, animations could have been better and the combat felt a little janky. | graphics/negative; gameplay/negative |
| Tony Hawks Pro Skater HD | 3- u cant change moves while u rail or lip .... 4- same thing for manuals.... this game is too old + its harder than what we used to play , more like project 8 , u cant get high combo ... | gameplay/negative; difficulty/negative |
| Test Drive Unlimited 2 | Other glaring faults I consider are that the voice acting leaves a LOT to be desired, the motivation of the characters are extremly questionable, the amount of customization is lacking in variety, & finally the 'unlocking' scheme is laughable. | audio/negative; story/negative; content/negative; gameplay/negative |
| Toren | Press Button to win. | gameplay/negative |
| Company of Heroes 2 | I recommend this game to advanced RTS players who don't care about losing, but casuals, please stay away from it ! | overall/neutral |
| Company of Heroes 2 | My favorite RTS, i still havent gotten around to playing campaign. | overall/positive |
| Company of Heroes 2 | Neg.: | [] |
| Alien Rage - Unlimited | Go slow and always have a fresh clip. | [] |

## System C error analysis (test)

448 of 851 test sentences (52.6%) have at least one error. Sentences per error type (a sentence can have several):

| error type | sentences |
|---|---|
| missed aspect | 259 |
| extra aspect | 183 |
| wrong polarity | 123 |
| no labels on an aspect sentence | 108 |
| labels on a no-aspect sentence | 49 |

Aspect errors (count of aspect instances):

| aspect | missed (FN) | extra (FP) |
|---|---|---|
| gameplay | 66 | 59 |
| story | 10 | 12 |
| graphics | 6 | 6 |
| audio | 3 | 6 |
| performance | 21 | 24 |
| price | 7 | 9 |
| multiplayer | 13 | 7 |
| difficulty | 11 | 13 |
| content | 17 | 17 |
| overall | 70 | 38 |
| other | 51 | 5 |

Polarity confusion on aspects found by both (rows gold, columns predicted):

| gold \ pred | positive | negative | neutral |
|---|---|---|---|
| positive | 241 | 16 | 11 |
| negative | 27 | 245 | 15 |
| neutral | 23 | 36 | 28 |

Per game (test):

| game | sentences | pair F1 | aspect F1 |
|---|---|---|---|
| Sniper: Ghost Warrior 2 | 392 | 0.592 | 0.744 |
| Two Worlds II HD | 459 | 0.580 | 0.721 |

Example errors (random, seed 42):

| sentence | gold | predicted | error |
|---|---|---|---|
| but he didnt die , he started running | [] | performance/negative | extra aspect, labels on a no-aspect sentence |
| and were programed by a class of kindergardeners who were all in a coma for 90% of their lives | gameplay/negative | [] | missed aspect, no labels on an aspect sentence |
| you can learn to play a whole variety of instruments as well as finding | content/positive | gameplay/positive | extra aspect, missed aspect |
| Should stick with what works. | gameplay/negative | [] | missed aspect, no labels on an aspect sentence |
| As the game progresses I found all the creatures still gave decent challenges, there was no level of gear or skills I obtained that made me one shot all the mobs. | difficulty/positive | gameplay/neutral; difficulty/neutral | extra aspect, wrong polarity |
| Graphics are 'Ok' at best Animations are complete ♥♥♥♥, | graphics/negative | graphics/positive | wrong polarity |
| It's pretty like The Witcher with great quests and adventures. | gameplay/positive | gameplay/positive; story/positive | extra aspect |
| It is ♥♥♥♥ing unplayable, I really really wanted to play this game, but instead the only thing I get is a game with no sound, the cut scenes are in slow motion, and sometimes the enemy wont even attack me. | performance/negative | audio/negative; performance/negative | extra aspect |
| Amazingly fun sniper experience. | overall/positive | gameplay/positive; overall/positive | extra aspect |
| gameplay 101/10 story- last of us level sound- 10/5 graphics-bf4 level | gameplay/positive; story/positive; audio/positive; graphics/positive | gameplay/positive; story/positive; graphics/positive; audio/positive; content/positive | extra aspect |
| It even goes up to locking your view so you cannot stray aside too much when aiming or using binnoculars. | gameplay/negative | gameplay/positive | wrong polarity |
| It's just too dull and boring. | overall/negative | gameplay/negative | extra aspect, missed aspect |
