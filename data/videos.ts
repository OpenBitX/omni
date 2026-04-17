export type Vibe =
  | "lonely_road"
  | "ocean_wide"
  | "autumn_lake"
  | "misty_mountain"
  | "desert_sky"
  | "city_nightfall"
  | "forest_path"
  | "coastal_wind";

export type SceneVideo = {
  id: string;
  src: string;
  poster?: string;
  title: string;
  vibe: Vibe;
  location: string;
};

// Drop five loopable, portrait (9:16) .mp4 files into /public/videos/
// and they will appear in the feed. The filenames here are suggestions —
// rename your files to match, or edit this list.
export const VIDEOS: SceneVideo[] = [
  {
    id: "sayram",
    src: "/videos/sayram.mp4",
    title: "Autumn of Sayram Lake",
    vibe: "autumn_lake",
    location: "Sayram Lake, Xinjiang",
  },
  {
    id: "pacific",
    src: "/videos/pacific.mp4",
    title: "The Coast at Dusk",
    vibe: "coastal_wind",
    location: "Pacific Coast Highway",
  },
  {
    id: "dolomites",
    src: "/videos/dolomites.mp4",
    title: "Mountains Before the Weather",
    vibe: "misty_mountain",
    location: "Dolomites, Italy",
  },
  {
    id: "mojave",
    src: "/videos/mojave.mp4",
    title: "An Empty Road at Noon",
    vibe: "lonely_road",
    location: "Mojave, California",
  },
  {
    id: "tokyo",
    src: "/videos/tokyo.mp4",
    title: "After the Last Train",
    vibe: "city_nightfall",
    location: "Shinjuku, Tokyo",
  },
];
