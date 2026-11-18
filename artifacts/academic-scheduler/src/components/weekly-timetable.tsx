import { Check, FlaskConical, MapPin, Users } from 'lucide-react';

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const SLOTS = [
  '8:00–9:00', '9:00–10:00', '10:00–11:00', '11:00–12:00',
  '12:00–1:00', '1:00–2:00', '2:00–3:00', '3:00–4:00',
  '4:00–5:00', '5:00–6:00',
];

type TimetableSession = {
  id: number;
  day: string;
  time: string;
  subject: string;
  faculty: string;
  room: string;
  division: string;
  classType?: string | null;
  batch?: string | null;
};

function getStartTime(value: string) {
  const start = value.match(/(\d{1,2}):(\d{2})/);
  return start ? `${Number(start[1])}:${start[2]}` : '';
}

function getSlotIndex(value: string) {
  const start = getStartTime(value);
  return SLOTS.findIndex((slot) => slot.split('–')[0] === start);
}

function classTone(classType: string) {
  if (classType.toLowerCase() === 'lab') return 'border-orange-200 bg-orange-100 text-orange-800';
  if (classType.toLowerCase() === 'tutorial') return 'border-violet-200 bg-violet-100 text-violet-800';
  return 'border-blue-200 bg-blue-100 text-blue-800';
}

export function WeeklyTimetable({
  sessions,
  audience,
}: {
  sessions: TimetableSession[];
  audience: 'student' | 'faculty';
}) {
  const sessionsAt = (day: string, slot: string) => {
    const slotIndex = SLOTS.indexOf(slot);
    return sessions.filter((session) => {
      if (session.day !== day) return false;
      const startIndex = getSlotIndex(session.time);
      const isLab = session.classType?.toLowerCase() === 'lab';
      return startIndex >= 0 && (startIndex === slotIndex || (isLab && startIndex === slotIndex - 1));
    });
  };

  return (
    <section className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm" data-testid={`${audience}-weekly-timetable`}>
      <div className="overflow-x-auto">
        <div className="min-w-[1050px]">
          <div className="grid grid-cols-[100px_repeat(6,minmax(155px,1fr))] border-b border-border bg-muted/40">
            <div className="p-3.5 font-mono text-[9px] uppercase tracking-[.14em] text-muted-foreground">Time</div>
            {DAYS.map((day) => (
              <div key={day} className="border-l border-border px-3.5 py-3">
                <p className="font-mono text-[9px] uppercase tracking-[.14em] text-muted-foreground">{day.slice(0, 3)}</p>
                <p className="mt-1 text-[11px] font-extrabold">{day}</p>
              </div>
            ))}
          </div>

          {SLOTS.map((slot) => (
            <div key={slot} className="grid grid-cols-[100px_repeat(6,minmax(155px,1fr))]">
              <div className="min-h-[105px] border-b border-border p-3 font-mono text-[9px] text-muted-foreground">{slot}</div>
              {DAYS.map((day) => (
                <div key={`${day}-${slot}`} className="min-h-[105px] border-b border-l border-border p-1.5">
                  {sessionsAt(day, slot).map((session) => {
                    const classType = session.classType || 'Theory';
                    const isLab = classType.toLowerCase() === 'lab';
                    const startIndex = getSlotIndex(session.time);
                    const isContinuation = isLab && startIndex >= 0 && startIndex < SLOTS.indexOf(slot);
                    const isShared = session.division.startsWith('__SHARED_MLOPS__');
                    return (
                      <article key={session.id} className="mb-1.5 rounded-xl border border-blue-200 bg-slate-50/90 p-2.5 text-slate-800">
                        <div className="mb-1.5 flex flex-wrap gap-1">
                          <span className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 font-mono text-[8px] font-bold uppercase tracking-[.08em] ${classTone(classType)}`}>
                            {isLab ? <FlaskConical size={9} /> : <Check size={9} />}
                            {isContinuation ? 'Lab · continued' : classType}
                          </span>
                          {session.batch && <span className="rounded-full bg-slate-200 px-1.5 py-0.5 font-mono text-[8px] font-bold uppercase tracking-[.08em] text-slate-600">{session.batch}</span>}
                        </div>
                        <p className="text-[10px] font-extrabold leading-[1.35]">{session.subject}</p>
                        {isShared && <p className="mt-0.5 font-mono text-[8px] font-bold uppercase tracking-wide text-primary">Shared MLOPS cohort</p>}
                        {isLab && <p className="mt-1 font-mono text-[8px] text-muted-foreground">{isContinuation ? `Continues · ${session.time}` : session.time}</p>}
                        <p className="mt-1 flex items-start gap-1 text-[9px] leading-3 text-muted-foreground">
                          <Users size={10} className="mt-0.5 shrink-0" />
                          {audience === 'student' ? session.faculty : isShared ? 'Shared MLOPS cohort' : `Division ${session.division.split(' · ')[0]}`}
                        </p>
                        <p className="mt-0.5 flex items-center gap-1 text-[9px] text-muted-foreground"><MapPin size={9} /> {session.room === 'Online' ? 'Online' : `Room ${session.room}`}</p>
                      </article>
                    );
                  })}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
