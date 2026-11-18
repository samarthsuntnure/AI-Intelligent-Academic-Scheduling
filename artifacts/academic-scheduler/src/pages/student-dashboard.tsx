import { useEffect, useState } from 'react';
import { Users } from 'lucide-react';
import { PageIntro, LoadingState, ErrorState, EmptyState } from '@/components/primitives';
import { WeeklyTimetable } from '@/components/weekly-timetable';
import { useAuth } from '@/lib/auth-context';

type Session = { id: number; day: string; time: string; subject: string; faculty: string; room: string; division: string; classType?: string; batch?: string | null; batchSize?: number | null };
type StudentSchedule = {
  student: { name: string; division: string; rollNo: number; prn: string | null };
  sessions: Session[];
  assignedFaculty: string[];
};

const DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

export default function StudentDashboard() {
  const { user } = useAuth();
  const [data, setData] = useState<StudentSchedule | null>(null);
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');

  async function load() {
    setState('loading');
    try {
      const res = await fetch('/api/me/student-schedule', { credentials: 'include' });
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
        eyebrow="My timetable"
        title={`Welcome, ${user?.name ?? 'there'}`}
        description={data ? `Division ${data.student.division} · Roll No. ${data.student.rollNo}` : 'Your class schedule and assigned faculty.'}
      />
      {state === 'loading' && <LoadingState rows={5} />}
      {state === 'error' && <ErrorState onRetry={load} message="Could not load your timetable." />}
      {state === 'ready' && (
        <>
          <div className="mb-6 rounded-2xl border border-border bg-card p-5" data-testid="card-faculty">
            <p className="mb-2 flex items-center gap-2 font-mono text-[10px] font-medium uppercase tracking-[.16em] text-muted-foreground">
              <Users size={13} /> Assigned faculty
            </p>
            {data && data.assignedFaculty.length > 0 ? (
              <div className="flex flex-wrap gap-x-5 gap-y-1">
                {data.assignedFaculty.map((name) => (
                  <p key={name} className="text-sm font-bold">{name}</p>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Not assigned yet.</p>
            )}
          </div>

          {sessions.length === 0 ? (
            <EmptyState title="No sessions scheduled yet" description="Once the admin office generates a timetable, your classes will show up here." />
          ) : <WeeklyTimetable sessions={sessions} audience="student" />}
        </>
      )}
    </div>
  );
}
