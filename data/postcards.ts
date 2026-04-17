import type { Vibe } from "./videos";

export type Postcard = {
  id: string;
  vibe: Vibe;
  text: string;
  signedBy?: string;
  at?: string;
};

// 50 seeded "unmailed postcards" — short, anonymous, emotionally specific.
// Written as if left behind by strangers who once stood here.
export const POSTCARDS: Postcard[] = [
  // lonely_road
  { id: "lr-01", vibe: "lonely_road", text: "I drove this road to escape a version of myself. I didn't come back with a different one. But I came back.", signedBy: "a passenger, no one's" },
  { id: "lr-02", vibe: "lonely_road", text: "There was no music. Just the wind against the side mirror, asking the same question for three hundred miles.", signedBy: "27, last August" },
  { id: "lr-03", vibe: "lonely_road", text: "I told my mother I was fine. Then I pulled over here and wasn't.", signedBy: "someone's kid" },
  { id: "lr-04", vibe: "lonely_road", text: "If you're reading this and you nearly turned around an hour ago — keep going.", signedBy: "— already forgot your name" },
  { id: "lr-05", vibe: "lonely_road", text: "The road doesn't care where you're going. I found that comforting.", signedBy: "no return address" },
  { id: "lr-06", vibe: "lonely_road", text: "I used to think solitude was a failure. Out here it feels like permission." },
  { id: "lr-07", vibe: "lonely_road", text: "Put on Qi Li Xiang. The wind is yours alone.", signedBy: "someone older than you think" },

  // ocean_wide
  { id: "ow-01", vibe: "ocean_wide", text: "I came here to cry. Ended up just listening. Turns out the ocean doesn't need my help.", signedBy: "anonymous, Tuesday" },
  { id: "ow-02", vibe: "ocean_wide", text: "My father loved this view. He never made it here. I brought him anyway." },
  { id: "ow-03", vibe: "ocean_wide", text: "I said things out loud I've never written down. The sea took them. No questions.", signedBy: "— 31" },
  { id: "ow-04", vibe: "ocean_wide", text: "Everything that feels urgent at home feels like a rumor here." },
  { id: "ow-05", vibe: "ocean_wide", text: "I held a stranger's hand for forty seconds. Neither of us said anything about it after.", signedBy: "briefly brave" },
  { id: "ow-06", vibe: "ocean_wide", text: "The tide did what my therapist has been asking me to do for two years. Came. Went. Came back anyway." },

  // autumn_lake
  { id: "al-01", vibe: "autumn_lake", text: "The water was so still I forgot to breathe. My reflection looked older than I feel inside.", signedBy: "a tourist, maybe" },
  { id: "al-02", vibe: "autumn_lake", text: "I was supposed to be here with someone. I'm glad I wasn't. That's a new sentence for me." },
  { id: "al-03", vibe: "autumn_lake", text: "Autumn has a way of making you nostalgic for things that haven't happened yet.", signedBy: "— 24, fond of long walks" },
  { id: "al-04", vibe: "autumn_lake", text: "I sat on the shore and made a list of the people I still owe an apology. The list was shorter than I expected." },
  { id: "al-05", vibe: "autumn_lake", text: "This is the autumn of Sayram Lake. The wind is yours alone.", signedBy: "passed through once" },
  { id: "al-06", vibe: "autumn_lake", text: "The leaves weren't falling. They were leaving. There's a difference." },

  // misty_mountain
  { id: "mm-01", vibe: "misty_mountain", text: "The clouds arrived like they'd been waiting for me to slow down. They had.", signedBy: "a woman in boots" },
  { id: "mm-02", vibe: "misty_mountain", text: "I stopped being afraid of being small up here. The mountains don't measure you." },
  { id: "mm-03", vibe: "misty_mountain", text: "I deleted his number on the ridge. The wind took it downhill. Don't look for it." },
  { id: "mm-04", vibe: "misty_mountain", text: "There's something the altitude does to regret. It thins it." },
  { id: "mm-05", vibe: "misty_mountain", text: "My guide didn't say a word for an hour. It was the kindest conversation I've had all year.", signedBy: "March, early thaw" },
  { id: "mm-06", vibe: "misty_mountain", text: "The fog hid the valley. The valley was still there. I'm trying to remember that." },

  // desert_sky
  { id: "ds-01", vibe: "desert_sky", text: "The stars out here made my problems feel unreasonable in the best way.", signedBy: "— from Phoenix, sort of" },
  { id: "ds-02", vibe: "desert_sky", text: "I came to see Saturn. Saw my own thoughts instead. Fair trade." },
  { id: "ds-03", vibe: "desert_sky", text: "The silence had weight. I put my grief down in it.", signedBy: "widow, not saying for how long" },
  { id: "ds-04", vibe: "desert_sky", text: "The desert doesn't ask why you came. That's the whole gift." },
  { id: "ds-05", vibe: "desert_sky", text: "I kept looking for a constellation I invented when I was twelve. Still haven't found it." },

  // city_nightfall
  { id: "cn-01", vibe: "city_nightfall", text: "I walked past his old apartment. The light was on. Someone else's light now.", signedBy: "not bitter, just observing" },
  { id: "cn-02", vibe: "city_nightfall", text: "You can be alone in a crowd and feel held by it. I've never written that down before." },
  { id: "cn-03", vibe: "city_nightfall", text: "The last train goes whether you're on it or not. That's the lesson of this city." },
  { id: "cn-04", vibe: "city_nightfall", text: "I ate a bowl of noodles at 1am next to a man in a perfect suit. Neither of us were doing well." },
  { id: "cn-05", vibe: "city_nightfall", text: "Tokyo in the rain is medicine. I have no science to back this up.", signedBy: "— 29, soaked through" },
  { id: "cn-06", vibe: "city_nightfall", text: "I was lonely for years in my hometown. I'm lonely here too, but differently. Better, I think." },

  // forest_path
  { id: "fp-01", vibe: "forest_path", text: "The birds had opinions. None of them were about me. It was such a relief." },
  { id: "fp-02", vibe: "forest_path", text: "I kept walking past the turnaround point. My body wanted to keep going even when I didn't.", signedBy: "will be late to dinner" },
  { id: "fp-03", vibe: "forest_path", text: "The light through the leaves was the exact color of something I haven't felt since I was nine." },
  { id: "fp-04", vibe: "forest_path", text: "I left a small stone on the path. If you pick it up, replace it with one of yours." },
  { id: "fp-05", vibe: "forest_path", text: "I said his name out loud for the first time in months. The trees didn't flinch." },

  // coastal_wind
  { id: "cw-01", vibe: "coastal_wind", text: "I told the wind about my father. It kept moving. I loved that.", signedBy: "recently someone's orphan" },
  { id: "cw-02", vibe: "coastal_wind", text: "Sunsets are embarrassing, aren't they. Corny. Unavoidable. I cried anyway." },
  { id: "cw-03", vibe: "coastal_wind", text: "The gulls took my sandwich and my last shred of pride. Worth it." },
  { id: "cw-04", vibe: "coastal_wind", text: "I kissed someone on this overlook eleven years ago. I'm here alone today. Both were good.", signedBy: "the girl in the green jacket, older now" },
  { id: "cw-05", vibe: "coastal_wind", text: "The horizon does something no therapist has managed: it stops the sentence I keep re-writing in my head." },
  { id: "cw-06", vibe: "coastal_wind", text: "I walked until the road ran out. I will tell no one what I decided there." },
];

export function postcardsForVibe(vibe: Vibe, n = 4): Postcard[] {
  const pool = POSTCARDS.filter((p) => p.vibe === vibe);
  const shuffled = [...pool].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, n);
}

export function totalForVibe(vibe: Vibe): number {
  // Inflate the seed count so early users feel part of a crowd.
  return POSTCARDS.filter((p) => p.vibe === vibe).length * 43 + 81;
}
