# Database Migrations with Alembic

This project uses Alembic for database schema migrations. This allows you to update your database schema when the code changes.

## Running Migrations

### For SQLite

```bash
source aiSafetyEnv/bin/activate
alembic -x database_url=sqlite:///./ai_monitor.db upgrade head
```

### For PostgreSQL

```bash
source aiSafetyEnv/bin/activate
alembic -x database_url=postgresql+psycopg://myuser:mypassword@localhost:5433/mydb upgrade head
```

Or set the URL in `alembic.ini` and run:
```bash
alembic upgrade head
```

## Common Commands

### Check Current Migration Status

```bash
alembic -x database_url=sqlite:///./ai_monitor.db current
```

### View Migration History

```bash
alembic history
```

### Create a New Migration

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "description of changes"

# Or create empty migration
alembic revision -m "description of changes"
```

### Rollback Migration

```bash
# Rollback one step
alembic -x database_url=sqlite:///./ai_monitor.db downgrade -1

# Rollback to specific revision
alembic -x database_url=sqlite:///./ai_monitor.db downgrade f13c97550b8f
```

### Stamp Database (Mark as migrated without running)

If you have an existing database that matches a migration state:

```bash
alembic -x database_url=sqlite:///./ai_monitor.db stamp head
```

## Migration Files

Migrations are stored in `migrations/versions/`. Each migration file contains:
- `upgrade()`: Function to apply the migration
- `downgrade()`: Function to rollback the migration

## Important Notes

1. **Always backup your database** before running migrations in production
2. **Test migrations** on a copy of production data first
3. **SQLite limitations**: SQLite has limited ALTER TABLE support. Some operations (like changing NOT NULL constraints) may not work and need to be handled differently
4. **PostgreSQL vs SQLite**: Some migrations may need database-specific logic (see `391610980da5_add_pipeline_columns.py` for an example)

## Current Migrations

1. `f13c97550b8f` - Initial migration: add prompt_payload to moderation_results
2. `391610980da5` - Add pipeline columns: input/output classification, answer generation fields

## Troubleshooting

### "Table already exists" errors

If you have an existing database that wasn't created with Alembic:
1. Stamp it to the appropriate revision: `alembic stamp <revision>`
2. Then run migrations: `alembic upgrade head`

### "Column already exists" errors

The column may have been added manually. You can either:
1. Remove the column manually and re-run the migration
2. Modify the migration to check if the column exists before adding it

### SQLite ALTER TABLE limitations

SQLite doesn't support:
- Changing column types
- Changing NOT NULL constraints
- Dropping columns (requires table recreation)

Workarounds are implemented in migrations where needed.

