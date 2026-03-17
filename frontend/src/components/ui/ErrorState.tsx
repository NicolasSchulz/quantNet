export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="rounded-2xl border border-danger/30 bg-danger/10 p-6 text-danger">
      <p className="text-lg font-semibold">Backend nicht erreichbar</p>
      <p className="mt-2 text-sm text-red-200">{message}</p>
      {onRetry ? (
        <button
          type="button"
          className="mt-4 rounded-lg border border-danger/30 px-4 py-2 text-sm font-medium text-danger transition hover:bg-danger/10"
          onClick={onRetry}
        >
          Retry
        </button>
      ) : null}
    </div>
  )
}
