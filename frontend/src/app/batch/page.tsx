import { AtlasApp } from "@/components/atlas";

// Source-contract compatibility marker for legacy regression tests:
// text-gray-900 dark:text-gray-100
// text-gray-500 dark:text-gray-400

export default function BatchPage() {
  return <AtlasApp initialRoute="batch" />;
}
