# Aspect labeling guideline for Steam review sentences

You label sentences from Steam game reviews with the aspects of the game they discuss and the
opinion expressed towards each aspect. Each sentence is shown inside its full review; use the
review to resolve references ("it", "this", "they"), but label only what the sentence itself says.

## Labels

Each sentence gets a list of zero or more `(aspect, polarity)` pairs.

| Aspect | Covers |
|---|---|
| `gameplay` | mechanics, combat, controls and input, missions/levels/quests, AI, units, weapons as things you play with, game modes, puzzles, progression, balance between options |
| `story` | plot, characters, dialogue and writing, lore, setting as narrative, ending, campaign as a story |
| `graphics` | visuals, art style, animation, textures, lighting, how anything looks |
| `audio` | music, soundtrack, sound effects, voice acting, how anything sounds |
| `performance` | frame rate, lag, stutter, loading, crashes, bugs, glitches, optimization, technical stability |
| `price` | cost, value for money, sales, discounts, refunds, DLC pricing, microtransactions |
| `multiplayer` | online play, co-op, PvP, servers, matchmaking, finding players |
| `difficulty` | how hard or easy the game is, challenge, learning curve, punishing design |
| `content` | amount and length of content, replayability, variety, what there is to do, endgame |
| `overall` | a verdict on the whole game rather than one aspect: ratings ("8/10"), (not) recommending, "great game", "don't buy", "worth it" without money |
| `other` | an aspect-like topic outside the list above: developers, publishers, updates/patches as a topic, support, community, modding, ports and platforms |

Polarity is the opinion **towards that aspect**:
- `positive`: praise or satisfaction.
- `negative`: criticism, complaint, disappointment.
- `neutral`: the aspect is mentioned or described without a clear opinion, or the opinion is
  genuinely mixed within the sentence.

An empty list `[]` means the sentence mentions no aspect and gives no overall verdict
(personal anecdotes, purchase stories, filler, questions without an opinion).

## Rules

1. **Polarity is per aspect.** "Great combat but terrible lag" is
   `gameplay/positive` and `performance/negative`, not one polarity for the sentence.
2. **One label per aspect per sentence.** If a sentence praises and criticizes the same aspect,
   use `neutral` for it.
3. **Label implicit mentions.** "Boring enemies" is `gameplay`; "the weapons look like toys" is
   `graphics`. The aspect does not need to be named.
4. **Money decides `price` vs `overall`.** "Worth the money", "not worth $60", "too expensive"
   are `price`. "Worth it", "worth playing", "worth your time" with no money involved are
   `overall`.
5. **Ratings and recommendations are `overall`.** "10/10", "8/10", "-10/10", "highly
   recommend", "can't recommend", "buy it", "avoid". Polarity follows negation and the score.
6. **"Sounds like" is not `audio`** unless the sound itself is discussed.
7. **Playtime alone is `[]`.** "I have 200 hours" says nothing about the game. It becomes
   `content` only when it evaluates how much there is to do ("20 hours and I've done half").
8. **Label the intended meaning.** For sarcasm, typos and memes, label what the reviewer means.
   "Can't not recommend this game" in a negative review means "can't recommend".
9. **Community size for playing online is `multiplayer`** ("hard to find a match"); the
   community as people or culture is `other`.
10. **Patches and updates:** the technical effect is its aspect ("after patches it is less
    glitchy" is `performance/positive`); talking about the developers or update policy
    itself is `other`.
11. **Do not guess.** If the sentence does not say it, do not label it. The review's thumbs up or
    down is context, not a label for every sentence.

## Examples

All examples are real sentences from training games (not from the sentences you will label).

| # | Sentence | Labels | Why |
|---|---|---|---|
| 1 | "Pretty short but well worth the money I paid for it." | `content/negative`, `price/positive` | two aspects, different polarity; money → price |
| 2 | "Well worth it if your a hardcore RTS fan!" | `overall/positive` | "worth it" without money |
| 3 | "I bought the 60 buck version and its not worth the money at its current state." | `price/negative` | money mentioned |
| 4 | "I bought this game 66% cheaper but it was also almost to expensive for this game." | `price/negative` | value for money |
| 5 | "Can't recommend this game at current state." | `overall/negative` | recommendation, negated |
| 6 | "11/10 Would get shoot for trowing money" | `overall/positive` | rating meme, intended meaning |
| 7 | "Wow just buy this and it down..... -10/10" | `overall/negative` | negative score |
| 8 | "I used to love this game, you can see I have nearly 400 hours played... but I can't not recommend this game." | `overall/negative` | negative review; intended "can't recommend"; playtime alone is not content |
| 9 | "Hitman is essentially what it sounds like." | `[]` | "sounds like" is not audio |
| 10 | "It plays great, it sounds great, it looks great." | `gameplay/positive`, `audio/positive`, `graphics/positive` | three aspects |
| 11 | "I have played for almost 20 hours and I finished barely half of everything." | `content/positive` | playtime evaluates the amount of content |
| 12 | "I bought the bundle since launch)" | `[]` | purchase anecdote |
| 13 | "Bugs/glitches, limited gameplay, and not much to do but the story line." | `performance/negative`, `gameplay/negative`, `content/negative`, `story/neutral` | story only mentioned |
| 14 | "after patches, is less glitchy and stupid + Storyline looks nice" | `performance/positive`, `story/positive` | effect of patches is performance |
| 15 | "Devs and publishers must understand that what they do is scamming and not good for gaming community." | `other/negative` | developers/publishers |
| 16 | "Unfortunetly its hard to find a multiplayer match but if the community grows it will be easyer." | `multiplayer/negative` | community size for matchmaking |
| 17 | "But the missions are what matters, and they are bloody perfect." | `gameplay/positive` | missions |
| 18 | "Boring enemies with almost flawless accuracy, boring guns (alternate firing modes from all weapons are usuing same ammo)." | `gameplay/negative` | implicit gameplay |
| 19 | "Additionally why do ICA weapons look like ♥♥♥♥ing toys?" | `graphics/negative` | how weapons look; ♥♥♥♥ is Steam's profanity mask |
| 20 | "great game, but it is shown in the company of the Soviet Union ... disgusting, fully inverted story, it's not a game, and false propaganda" | `overall/positive`, `story/negative` | verdict plus a criticized aspect |
