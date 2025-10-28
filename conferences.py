from base_etl import BaseCollegeBasketballETL
from typing import List, Dict


class ConferencesETL(BaseCollegeBasketballETL):
    """ETL for college basketball conferences data"""
    
    def create_table(self, cursor):
        """Create the conferences table if it doesn't exist"""
        create_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='conferences' AND xtype='U')
        CREATE TABLE conferences (
            id INT PRIMARY KEY,
            sourceId NVARCHAR(50),
            name NVARCHAR(200),
            abbreviation NVARCHAR(50),
            shortName NVARCHAR(100),
            
            INDEX IX_Conferences_Abbreviation (abbreviation),
            INDEX IX_Conferences_Name (name)
        )
        """
        cursor.execute(create_table_sql)
        print("Table 'conferences' verified/created")
    
    def fetch_conferences(self) -> List[Dict]:
        """
        Fetch all conferences
        
        Returns:
            List of conference dictionaries
        """
        endpoint = "/conferences"
        return self.fetch_api(endpoint)
    
    def insert_conferences(self, conferences: List[Dict], cursor) -> int:
        """
        Insert conferences into the database
        
        Args:
            conferences: List of conference dictionaries
            cursor: Database cursor
            
        Returns:
            Number of records inserted/updated
        """
        insert_sql = """
        MERGE conferences AS target
        USING (SELECT ? AS id) AS source
        ON target.id = source.id
        WHEN MATCHED THEN
            UPDATE SET
                sourceId = ?,
                name = ?,
                abbreviation = ?,
                shortName = ?
        WHEN NOT MATCHED THEN
            INSERT (id, sourceId, name, abbreviation, shortName)
            VALUES (?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for conf in conferences:
            try:
                params = (
                    # For MERGE condition
                    conf.get('id'),
                    # For UPDATE
                    conf.get('sourceId'),
                    conf.get('name'),
                    conf.get('abbreviation'),
                    conf.get('shortName'),
                    # For INSERT
                    conf.get('id'),
                    conf.get('sourceId'),
                    conf.get('name'),
                    conf.get('abbreviation'),
                    conf.get('shortName')
                )
                
                if self.execute_merge(cursor, insert_sql, params, 
                                     f"conference {conf.get('name')}"):
                    inserted += 1
                    
            except Exception as e:
                print(f"Error processing conference {conf.get('name')}: {e}")
                continue
        
        return inserted
    
    def run_etl(self):
        """
        Run the complete ETL process for conferences
        
        This is a simple one-time fetch since conferences don't change by season
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Create table if needed
            self.create_table(cursor)
            conn.commit()
            
            print(f"\n{'='*60}")
            print(f"Fetching conferences...")
            print(f"{'='*60}")
            
            # Fetch conferences
            conferences = self.fetch_conferences()
            
            if not conferences:
                print("No conferences found")
                return
            
            print(f"Found {len(conferences)} conferences")
            
            # Insert conferences
            inserted = self.insert_conferences(conferences, cursor)
            conn.commit()
            
            print(f"\n{'='*60}")
            print(f"=== ETL Complete ===")
            print(f"{'='*60}")
            print(f"Total conferences inserted/updated: {inserted}")
            
        except Exception as e:
            print(f"ETL process error: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()


# Example usage
if __name__ == "__main__":
    # Configuration
    API_KEY = "your_api_key_here"
    
    # MSSQL connection string
    DB_CONNECTION = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=your_server_name;"
        "DATABASE=your_database_name;"
        "UID=your_username;"
        "PWD=your_password"
    )
    
    # Initialize and run ETL
    etl = ConferencesETL(API_KEY, DB_CONNECTION)
    
    # Simple one-time fetch - no season parameters needed
    etl.run_etl()