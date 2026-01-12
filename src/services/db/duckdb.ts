import { type DuckDBConnection, DuckDBInstance } from '@duckdb/node-api';
import { appConfig } from '~/config';

const {
  data: { duckDb },
} = appConfig;

const duckDBThreads = 4;

// Lazy initialization to avoid running during SSR build
let duckDB: DuckDBInstance | null = null;
let initialized = false;

const getDuckDB = async (): Promise<DuckDBInstance> => {
  if (!duckDB) {
    duckDB = await DuckDBInstance.create(duckDb);
  }
  return duckDB;
};

const initializeExtensions = async (db: DuckDBConnection) => {
  if (initialized) return;

  try {
    await db.run(`
      INSTALL gsheets FROM community;
      INSTALL nanoarrow FROM community;

      LOAD gsheets;
      LOAD nanoarrow;

      SET threads=${duckDBThreads};
      SET temp_directory = '${duckDb}_duckdb_swap';
      SET memory_limit='128GB';
      SET max_temp_directory_size = '256GB';
    `);
    initialized = true;
  } catch (error) {
    // Log but don't fail - extensions may not be available in all environments
    console.warn('[DuckDB] Extension initialization warning:', error);
    initialized = true; // Don't retry
  }
};

export const dbConnect = async () => {
  const instance = await getDuckDB();
  return await instance.connect();
};

type GenericIdentityFn<T> = (db: DuckDBConnection) => Promise<T>;

export const connectAndClose = async <T>(
  func: GenericIdentityFn<T>,
): Promise<T> => {
  const db = await dbConnect();
  try {
    await initializeExtensions(db);
    const result = await func(db);
    return result;
  } catch (error) {
    console.error(error);
    throw error;
  } finally {
    db.disconnectSync();
    db.closeSync();
  }
};
