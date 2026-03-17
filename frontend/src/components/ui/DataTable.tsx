import type { ReactNode } from "react"

import { cn } from "../../utils/formatters"

export interface Column<T> {
  key: string
  header: string
  sortable?: boolean
  className?: string
  render: (row: T) => ReactNode
}

export function DataTable<T>({
  columns,
  rows,
  sortKey,
  sortDirection,
  onSort,
  onRowClick,
  emptyState,
}: {
  columns: Column<T>[]
  rows: T[]
  sortKey?: string
  sortDirection?: "asc" | "desc"
  onSort?: (key: string) => void
  onRowClick?: (row: T) => void
  emptyState: ReactNode
}) {
  if (rows.length === 0) {
    return <div className="rounded-2xl border border-dashed border-border bg-secondary p-10 text-center text-textSecondary">{emptyState}</div>
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-secondary shadow-card">
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-primary/60 text-xs uppercase tracking-wide text-textSecondary">
            <tr>
              {columns.map((column) => (
                <th key={column.key} className={cn("px-4 py-3 font-medium", column.className)}>
                  <button
                    type="button"
                    className={cn("flex items-center gap-2", column.sortable ? "hover:text-textPrimary" : "cursor-default")}
                    onClick={() => column.sortable && onSort?.(column.key)}
                  >
                    {column.header}
                    {sortKey === column.key ? <span>{sortDirection === "asc" ? "↑" : "↓"}</span> : null}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr
                key={index}
                className={cn(
                  "border-t border-border transition hover:bg-tertiary",
                  index % 2 === 0 ? "bg-secondary" : "bg-primary/30",
                  onRowClick ? "cursor-pointer" : "",
                )}
                onClick={() => onRowClick?.(row)}
              >
                {columns.map((column) => (
                  <td key={column.key} className={cn("px-4 py-3 align-middle text-textPrimary", column.className)}>
                    {column.render(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
