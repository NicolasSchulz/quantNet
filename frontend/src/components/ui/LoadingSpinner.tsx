export function LoadingSpinner({ label = "Loading..." }: { label?: string }) {
  return (
    <div className="flex min-h-[180px] items-center justify-center gap-3 text-sm text-textSecondary">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-border border-t-success" />
      <span>{label}</span>
    </div>
  )
}
