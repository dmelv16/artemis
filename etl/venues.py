from etl.base_etl import BaseCollegeBasketballETL
from typing import List, Dict


class VenuesETL(BaseCollegeBasketballETL):
    """ETL for college basketball venues data"""
    
    def create_table(self, cursor):
        """Create the venues table if it doesn't exist"""
        create_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='venues' AND xtype='U')
        CREATE TABLE venues (
            id INT PRIMARY KEY,
            sourceId NVARCHAR(50),
            name NVARCHAR(200),
            city NVARCHAR(100),
            state NVARCHAR(50),
            country NVARCHAR(100),
            
            INDEX IX_Venues_Name (name),
            INDEX IX_Venues_City (city),
            INDEX IX_Venues_State (state),
            INDEX IX_Venues_Country (country)
        )
        """
        cursor.execute(create_table_sql)
        print("Table 'venues' verified/created")
    
    def fetch_venues(self) -> List[Dict]:
        """
        Fetch all venues
        
        Returns:
            List of venue dictionaries
        """
        endpoint = "/venues"
        return self.fetch_api(endpoint)
    
    def insert_venues(self, venues: List[Dict], cursor) -> int:
        """
        Insert venues into the database
        
        Args:
            venues: List of venue dictionaries
            cursor: Database cursor
            
        Returns:
            Number of records inserted/updated
        """
        insert_sql = """
        MERGE venues AS target
        USING (SELECT ? AS id) AS source
        ON target.id = source.id
        WHEN MATCHED THEN
            UPDATE SET
                sourceId = ?,
                name = ?,
                city = ?,
                state = ?,
                country = ?
        WHEN NOT MATCHED THEN
            INSERT (id, sourceId, name, city, state, country)
            VALUES (?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for venue in venues:
            try:
                params = (
                    # For MERGE condition
                    venue.get('id'),
                    # For UPDATE
                    venue.get('sourceId'),
                    venue.get('name'),
                    venue.get('city'),
                    venue.get('state'),
                    venue.get('country'),
                    # For INSERT
                    venue.get('id'),
                    venue.get('sourceId'),
                    venue.get('name'),
                    venue.get('city'),
                    venue.get('state'),
                    venue.get('country')
                )
                
                if self.execute_merge(cursor, insert_sql, params, 
                                     f"venue {venue.get('name')}"):
                    inserted += 1
                    
            except Exception as e:
                print(f"Error processing venue {venue.get('name')}: {e}")
                continue
        
        return inserted
    
    def run_etl(self):
        """
        Run the complete ETL process for venues
        
        This is a simple one-time fetch since venues are reference data
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Create table if needed
            self.create_table(cursor)
            conn.commit()
            
            print(f"\n{'='*60}")
            print(f"Fetching venues...")
            print(f"{'='*60}")
            
            # Fetch venues
            venues = self.fetch_venues()
            
            if not venues:
                print("No venues found")
                return
            
            print(f"Found {len(venues)} venues")
            
            # Insert venues
            inserted = self.insert_venues(venues, cursor)
            conn.commit()
            
            print(f"\n{'='*60}")
            print(f"=== ETL Complete ===")
            print(f"{'='*60}")
            print(f"Total venues inserted/updated: {inserted}")
            
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
    etl = VenuesETL(API_KEY, DB_CONNECTION)
    
    # Simple one-time fetch - no parameters needed
    etl.run_etl()