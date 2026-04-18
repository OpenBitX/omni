import { Stack } from "expo-router";
import Tracker from "@/components/tracker";

export default function Index() {
  return (
    <>
      <Stack.Screen options={{ title: "Tracker" }} />
      <Tracker />
    </>
  );
}
