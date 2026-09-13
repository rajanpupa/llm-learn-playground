import { notFound } from "next/navigation";
import { LESSONS, getLesson } from "@/lib/lessons";
import { LESSON_COMPONENTS } from "@/components/lessons/registry";

export function generateStaticParams() {
  return LESSONS.map((l) => ({ slug: l.slug }));
}

export default async function LessonPage({
  params,
}: {
  params: Promise<{ slug: string }> | { slug: string };
}) {
  const { slug } = await params;
  const lesson = getLesson(slug);
  if (!lesson) notFound();
  const Component = LESSON_COMPONENTS[slug];
  if (!Component) notFound();
  return <Component />;
}
