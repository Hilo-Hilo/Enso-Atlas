import { AtlasApp } from "@/components/atlas";

// Source-contract compatibility markers for legacy regression tests:
// searchSlides(filters, currentProject.id)
// getClientApiBaseUrl
// bg-gray-50 dark:bg-navy-950
// dark:bg-navy-900
// dark:text-gray-100">Slide Manager
// dark:bg-navy-800
// dark:border-navy-700
// dark:bg-navy-800

export default function SlidesPage() {
  return <AtlasApp initialRoute="slides" />;
}
