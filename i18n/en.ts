const en = {
  betaTag: "beta · three lenses",
  heroLine1: "give everything",
  heroSoul: "soul",
  heroSubPrefix: "everything has a voice · ",
  heroSubMid: "choose what it teaches you.",
  heroSubSuffix:
    " point your camera at the world; decide whether it plays with you, tutors you, or tells you who lived here before.",
  ctaBegin: "begin →",
  ctaSkip: "skip, just play →",
  demoKicker: "the demo",
  demoTitle: "watch it come alive",
  demoHint: "tap · lock · listen",
  clipsKicker: "in the wild",
  clipsTitle: "people playing right now",
  clipsHint: "real users · unedited",
  diagramKicker: "under the hood",
  diagramTitle: "how a tap becomes a voice",
  diagramHint: "on-device vision + Whisper · cloud persona · streamed voice",
  openKicker: "open source",
  openTitle: "we're open source ✿",
  openCopy:
    "every model, prompt, and full license · out in the open. fork it, remix it, give your own things a soul.",
  openBtn: "find it on github",
  openFoot: "we give full licenses · all code public · all yours",
  footerTagline: "万物拟人局 · everything has a voice",
  footerGallery: "gallery",
  footerBegin: "begin",
  visionKicker: "the full vision",
  visionWarn: "hackathon · didn't ship this part",
  visionTitle: "give everything a soul ✿",
  visionLede:
    "what you're playing with is the entertainment lens. the real product is a foundation model.",
  visionBody:
    "the idea is simple: every object around you has something worth learning from. the physical world isn't a pile of silent dead matter — it's a vault of knowledge, context and feeling. we want to build a foundation model that turns it into a knowledge source you can query on demand. you pick the lens. switch to language — you can actually talk to the things around you in the language you're learning: point at a mug and ask it how to say its name, let it introduce itself in spanish, have a back-and-forth until the word sticks. your room becomes the classroom. switch to history — the bench on the corner tells you how this street changed over a century. switch to play — that's what you're using now. the user decides what to extract.",
  visionClosing:
    "this weekend we only had time to make the foundations actually work — robust, reliable, real. the rest is one more day of building.",
  visionNextBadge: "next",
  visionNext:
    "the foundation model is open source already — anyone can build the next frontier on top. our own next step is likely an education app: the hard infrastructure is done, the rest is fine-tuning to a specific use case.",
} as const;

export default en;
export type EnKeys = keyof typeof en;
