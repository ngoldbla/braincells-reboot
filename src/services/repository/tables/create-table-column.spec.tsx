import { randomUUID } from 'node:crypto';

import { afterEach, describe, expect, it } from 'vitest';
import { connectAndClose } from '~/services/db/duckdb';
import { createDatasetTable } from './create-table';
import { createDatasetTableColumn } from './create-table-column';
import { deleteDatasetTable } from './delete-table';
import { getDatasetTableName } from './utils';

const dataset = {
  id: randomUUID(),
  name: 'test-dataset',
  createdBy: 'test-user',
};

afterEach(async () => {
  await deleteDatasetTable(dataset);
});

describe('createDatasetColumn', () => {
  it('creates a new dataset column ', async () => {
    const column = {
      id: randomUUID(),
      name: 'test-column',
      type: 'INTEGER',
    };

    await createDatasetTable({ dataset });

    await createDatasetTableColumn({
      dataset,
      column,
    });

    await connectAndClose(async (db) => {
      const result = await db.run(`DESCRIBE ${getDatasetTableName(dataset)}`);

      expect(await result.getRowObjects()).toEqual([
        {
          column_name: 'rowIdx',
          column_type: 'BIGINT',
          default: null,
          extra: null,
          key: null,
          null: 'YES',
        },
        {
          column_name: column.id,
          column_type: 'VARCHAR',
          null: 'YES',
          extra: null,
          default: null,
          key: null,
        },
      ]);
    });
  });
});
