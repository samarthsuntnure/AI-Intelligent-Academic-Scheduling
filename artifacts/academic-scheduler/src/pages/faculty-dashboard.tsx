import { useEffect, useState } from 'react';
import { PageIntro, LoadingState, ErrorState, EmptyState } from '@/components/primitives';
import { WeeklyTimetable } from '@/components/weekly-timetable';
import { useAuth } from '@/lib/auth-context';

type Session = { id: number; day: string; time: string; subject: string; faculty: string; room: string; division: string; classType?: string; batch?: string | null; batchSize?: number | null };
type FacultySchedule = { faculty: { name: string; department: string | null; maxWeeklyHours: number }; sessions: Session[] };

const DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

export default function FacultyDashboard() {
  const { user } = useAuth();
  const [data, setData] = useState<FacultySchedule | null>(null);
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');

  async function load() {
    setState('loading');
    try {
      const res = await fetch('/api/me/faculty-schedule', { credentials: 'include' });
      if (!res.ok) throw new Error('failed');
      setData(await res.json());
      setState('ready');
    } catch {
      setState('error');
    }
  }

  useEffect(() => {
    load();
  }, []);

  const sessions = [...(data?.sessions ?? [])].sort(
    (a, b) => DAY_ORDER.indexOf(a.day) - DAY_ORDER.indexOf(b.day) || a.time.localeCompare(b.time),
  );

  return (
    <div>
      <PageIntro
        eyebrow="My schedule"
        title={`Welcome, ${user?.name ?? 'there'}`}
        description={
          data
            ? `${sessions.length} session${sessions.length === 1 ? '' : 's'} this week · ${data.faculty.department ?? 'Faculty'} · up to ${data.faculty.maxWeeklyHours}h/week`
            : 'Your weekly teaching schedule.'
        }
      />
      {state === 'loading' && <LoadingState rows={5} />}
      {state === 'error' && <ErrorState onRetry={load} message="Could not load your schedule." />}
      {state === 'ready' && sessions.length === 0 && (
        <EmptyState title="No sessions scheduled yet" description="Once the admin office generates a timetable, your sessions will show up here." />
      )}
      {state === 'ready' && sessions.length > 0 && (
        <WeeklyTimetable sessions={sessions} audience="faculty" />
      )}
    </div>
  );
}
