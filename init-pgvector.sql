-- Create the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify the extension was created successfully
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
    ) THEN
        RAISE EXCEPTION 'pgvector extension was not created successfully';
    END IF;
    
    -- Test vector functionality
    CREATE TEMP TABLE test_vector (id int, vec vector(3));
    INSERT INTO test_vector VALUES (1, '[1,2,3]'::vector);
    DROP TABLE test_vector;
    
    RAISE NOTICE 'pgvector extension installed and working correctly';
END $$;