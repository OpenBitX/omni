import { useEffect, useRef, useState } from "react";
import { Animated, Easing, StyleSheet, Text, View } from "react-native";

// RN port of the browser speech bubble. Three states:
//   - thinking (dots)
//   - revealing (typewriter over shownCaption)
//   - exiting (fade + scale out, 260ms)
// Parent flips `caption` to null to dismiss; we hold the text through the
// exit animation so it doesn't snap to empty mid-fade.

const EXIT_MS = 260;
const REVEAL_MS_PER_CHAR = 24;
const MAX_REVEAL_MS = 1400;

type SpeechBubbleProps = {
  caption: string | null;
  thinking: boolean;
  speaking: boolean;
  maxWidth: number;
};

export function SpeechBubble({
  caption,
  thinking,
  speaking,
  maxWidth,
}: SpeechBubbleProps) {
  const [shownCaption, setShownCaption] = useState<string | null>(caption);
  const [phase, setPhase] = useState<"hidden" | "in" | "out">(
    caption || thinking ? "in" : "hidden"
  );
  const [revealedChars, setRevealedChars] = useState(0);
  const exitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const revealTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const scale = useRef(new Animated.Value(caption || thinking ? 1 : 0.4)).current;
  const opacity = useRef(new Animated.Value(caption || thinking ? 1 : 0)).current;

  useEffect(() => {
    if (caption) {
      setShownCaption((prev) => (prev === caption ? prev : caption));
      setPhase("in");
      if (exitTimerRef.current != null) {
        clearTimeout(exitTimerRef.current);
        exitTimerRef.current = null;
      }
      return;
    }
    if (thinking) {
      setPhase("in");
      if (exitTimerRef.current != null) {
        clearTimeout(exitTimerRef.current);
        exitTimerRef.current = null;
      }
      return;
    }
    if (phase === "in") {
      setPhase("out");
      exitTimerRef.current = setTimeout(() => {
        setPhase("hidden");
        setShownCaption(null);
        exitTimerRef.current = null;
      }, EXIT_MS);
    }
  }, [caption, thinking, phase]);

  useEffect(() => {
    if (phase === "in") {
      Animated.parallel([
        Animated.spring(scale, {
          toValue: 1,
          friction: 6,
          tension: 140,
          useNativeDriver: true,
        }),
        Animated.timing(opacity, {
          toValue: 1,
          duration: 220,
          useNativeDriver: true,
        }),
      ]).start();
    } else if (phase === "out") {
      Animated.parallel([
        Animated.timing(scale, {
          toValue: 0.6,
          duration: EXIT_MS,
          easing: Easing.in(Easing.cubic),
          useNativeDriver: true,
        }),
        Animated.timing(opacity, {
          toValue: 0,
          duration: EXIT_MS,
          easing: Easing.in(Easing.cubic),
          useNativeDriver: true,
        }),
      ]).start();
    }
  }, [phase, scale, opacity]);

  useEffect(() => {
    if (revealTimerRef.current != null) {
      clearInterval(revealTimerRef.current);
      revealTimerRef.current = null;
    }
    if (!shownCaption) {
      setRevealedChars(0);
      return;
    }
    setRevealedChars(0);
    const len = shownCaption.length;
    const perChar = Math.max(
      12,
      Math.min(REVEAL_MS_PER_CHAR, MAX_REVEAL_MS / Math.max(1, len))
    );
    revealTimerRef.current = setInterval(() => {
      setRevealedChars((n) => {
        const next = n + 1;
        if (next >= len && revealTimerRef.current != null) {
          clearInterval(revealTimerRef.current);
          revealTimerRef.current = null;
        }
        return next;
      });
    }, perChar);
    return () => {
      if (revealTimerRef.current != null) {
        clearInterval(revealTimerRef.current);
        revealTimerRef.current = null;
      }
    };
  }, [shownCaption]);

  useEffect(() => {
    return () => {
      if (exitTimerRef.current != null) clearTimeout(exitTimerRef.current);
      if (revealTimerRef.current != null) clearInterval(revealTimerRef.current);
    };
  }, []);

  if (phase === "hidden") return null;
  const isThinking = !shownCaption && thinking;

  return (
    <Animated.View
      style={[styles.root, { maxWidth, transform: [{ scale }], opacity }]}
      pointerEvents="none"
    >
      <View style={styles.bubble}>
        {isThinking ? (
          <ThinkingDots />
        ) : (
          <Text style={styles.caption} numberOfLines={0}>
            {shownCaption ? shownCaption.slice(0, revealedChars) : ""}
          </Text>
        )}
      </View>
      <View style={[styles.tail, speaking && styles.tailSpeaking]} />
    </Animated.View>
  );
}

function ThinkingDots() {
  const a = useRef(new Animated.Value(0.3)).current;
  const b = useRef(new Animated.Value(0.3)).current;
  const c = useRef(new Animated.Value(0.3)).current;
  useEffect(() => {
    const loop = (v: Animated.Value, delay: number) =>
      Animated.loop(
        Animated.sequence([
          Animated.timing(v, {
            toValue: 1,
            duration: 360,
            delay,
            useNativeDriver: true,
          }),
          Animated.timing(v, {
            toValue: 0.3,
            duration: 740,
            useNativeDriver: true,
          }),
        ])
      );
    const animA = loop(a, 0);
    const animB = loop(b, 150);
    const animC = loop(c, 300);
    animA.start();
    animB.start();
    animC.start();
    return () => {
      animA.stop();
      animB.stop();
      animC.stop();
    };
  }, [a, b, c]);
  return (
    <View style={styles.dots}>
      <Animated.View style={[styles.dot, { opacity: a }]} />
      <Animated.View style={[styles.dot, { opacity: b }]} />
      <Animated.View style={[styles.dot, { opacity: c }]} />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { position: "relative", alignSelf: "center" },
  bubble: {
    backgroundColor: "#ffffff",
    paddingHorizontal: 20,
    paddingVertical: 13,
    borderRadius: 26,
    borderWidth: 1,
    borderColor: "rgba(255,192,219,0.55)",
    shadowColor: "#ec4899",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.18,
    shadowRadius: 22,
    elevation: 5,
  },
  caption: {
    color: "#2a1540",
    fontSize: 15,
    lineHeight: 21,
    textAlign: "center",
    fontStyle: "italic",
  },
  dots: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingVertical: 3,
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: "rgba(236,72,153,0.8)",
  },
  tail: {
    position: "absolute",
    alignSelf: "center",
    bottom: -5,
    width: 12,
    height: 12,
    borderRadius: 3,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "rgba(255,192,219,0.55)",
    transform: [{ rotate: "45deg" }],
  },
  tailSpeaking: {},
});
