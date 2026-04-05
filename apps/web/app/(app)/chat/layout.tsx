/** Chat is a full-viewport column: scroll lives in the thread, not the shell. */
export default function ChatLayout({ children }: { children: React.ReactNode }) {
  return <div className="flex min-h-0 flex-1 flex-col overflow-hidden">{children}</div>
}
