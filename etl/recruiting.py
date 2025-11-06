from etl.base_etl import BaseCollegeBasketballETL
from typing import List, Dict


class RecruitingETL(BaseCollegeBasketballETL):
    """ETL for college basketball recruiting data"""
    
    def create_table(self, cursor):
        """Create the recruiting table if it doesn't exist"""
        create_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='recruiting' AND xtype='U')
        CREATE TABLE recruiting (
            id INT PRIMARY KEY,
            sourceId NVARCHAR(50),
            position NVARCHAR(10),
            schoolId INT,
            school NVARCHAR(200),
            
            -- Hometown details
            hometownCity NVARCHAR(100),
            hometownState NVARCHAR(50),
            hometownCountry NVARCHAR(100),
            hometownLatitude FLOAT,
            hometownLongitude FLOAT,
            hometownCountyFips NVARCHAR(10),
            
            -- Commitment details
            committedToId INT,
            committedToName NVARCHAR(200),
            committedToConference NVARCHAR(50),
            
            athleteId INT,
            year INT,
            name NVARCHAR(200),
            heightInches INT,
            weightPounds INT,
            stars INT,
            rating FLOAT,
            ranking INT,
            
            INDEX IX_Recruiting_Year (year),
            INDEX IX_Recruiting_AthleteId (athleteId),
            INDEX IX_Recruiting_CommittedToId (committedToId),
            INDEX IX_Recruiting_Ranking (ranking),
            INDEX IX_Recruiting_Stars (stars)
        )
        """
        cursor.execute(create_table_sql)
        print("Table 'recruiting' verified/created")
    
    def fetch_recruiting_year(self, year: int) -> List[Dict]:
        """
        Fetch recruiting data for a specific year
        
        Args:
            year: Recruiting class year
            
        Returns:
            List of recruit dictionaries
        """
        endpoint = "/recruiting/players"
        params = {"year": year}
        return self.fetch_api(endpoint, params)
    
    def insert_recruiting(self, recruits: List[Dict], cursor) -> int:
        """
        Insert recruiting data into the database
        
        Args:
            recruits: List of recruit dictionaries
            cursor: Database cursor
            
        Returns:
            Number of records inserted
        """
        insert_sql = """
        MERGE recruiting AS target
        USING (SELECT ? AS id) AS source
        ON target.id = source.id
        WHEN MATCHED THEN
            UPDATE SET
                sourceId = ?,
                position = ?,
                schoolId = ?,
                school = ?,
                hometownCity = ?,
                hometownState = ?,
                hometownCountry = ?,
                hometownLatitude = ?,
                hometownLongitude = ?,
                hometownCountyFips = ?,
                committedToId = ?,
                committedToName = ?,
                committedToConference = ?,
                athleteId = ?,
                year = ?,
                name = ?,
                heightInches = ?,
                weightPounds = ?,
                stars = ?,
                rating = ?,
                ranking = ?
        WHEN NOT MATCHED THEN
            INSERT (id, sourceId, position, schoolId, school,
                    hometownCity, hometownState, hometownCountry, hometownLatitude, 
                    hometownLongitude, hometownCountyFips,
                    committedToId, committedToName, committedToConference,
                    athleteId, year, name, heightInches, weightPounds, 
                    stars, rating, ranking)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for recruit in recruits:
            try:
                hometown = recruit.get('hometown', {})
                committed_to = recruit.get('committedTo', {})
                
                params = (
                    # For MERGE condition
                    recruit.get('id'),
                    # For UPDATE
                    recruit.get('sourceId'),
                    recruit.get('position'),
                    recruit.get('schoolId'),
                    recruit.get('school'),
                    hometown.get('city'),
                    hometown.get('state'),
                    hometown.get('country'),
                    hometown.get('latitude'),
                    hometown.get('longitude'),
                    hometown.get('countyFips'),
                    committed_to.get('id'),
                    committed_to.get('name'),
                    committed_to.get('conference'),
                    recruit.get('athleteId'),
                    recruit.get('year'),
                    recruit.get('name'),
                    recruit.get('heightInches'),
                    recruit.get('weightPounds'),
                    recruit.get('stars'),
                    recruit.get('rating'),
                    recruit.get('ranking'),
                    # For INSERT
                    recruit.get('id'),
                    recruit.get('sourceId'),
                    recruit.get('position'),
                    recruit.get('schoolId'),
                    recruit.get('school'),
                    hometown.get('city'),
                    hometown.get('state'),
                    hometown.get('country'),
                    hometown.get('latitude'),
                    hometown.get('longitude'),
                    hometown.get('countyFips'),
                    committed_to.get('id'),
                    committed_to.get('name'),
                    committed_to.get('conference'),
                    recruit.get('athleteId'),
                    recruit.get('year'),
                    recruit.get('name'),
                    recruit.get('heightInches'),
                    recruit.get('weightPounds'),
                    recruit.get('stars'),
                    recruit.get('rating'),
                    recruit.get('ranking')
                )
                
                if self.execute_merge(cursor, insert_sql, params, 
                                     f"recruit {recruit.get('name')} ({recruit.get('year')})"):
                    inserted += 1
                    
            except Exception as e:
                print(f"Error processing recruit {recruit.get('name')}: {e}")
                continue
        
        return inserted
    
    def run_etl(self, start_year: int = 2006, end_year: int = 2025):
        """
        Run the complete ETL process for recruiting data
        
        Args:
            start_year: Starting recruiting class year
            end_year: Ending recruiting class year
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Create table if needed
            self.create_table(cursor)
            conn.commit()
            
            total_inserted = 0
            
            # Process each recruiting class year
            for year in range(start_year, end_year + 1):
                print(f"\n{'='*60}")
                print(f"Processing recruiting class {year}...")
                print(f"{'='*60}")
                
                recruits = self.fetch_recruiting_year(year)
                
                if not recruits:
                    print(f"No recruiting data found for {year}")
                    self.rate_limit_sleep(0.5)
                    continue
                
                print(f"Found {len(recruits)} recruits for class {year}")
                
                inserted = self.insert_recruiting(recruits, cursor)
                conn.commit()
                total_inserted += inserted
                print(f"✓ Inserted/updated {inserted} recruits for class {year}")
                
                # Rate limiting
                self.rate_limit_sleep(0.5)
            
            print(f"\n{'='*60}")
            print(f"=== ETL Complete ===")
            print(f"{'='*60}")
            print(f"Total recruits inserted/updated: {total_inserted}")
            
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
    etl = RecruitingETL(API_KEY, DB_CONNECTION)
    
    # Process all recruiting classes from 2006-2025
    etl.run_etl(start_year=2006, end_year=2025)
    
    # Or process a single year
    # etl.run_etl(start_year=2024, end_year=2024)