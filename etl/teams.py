from etl.base_etl import BaseCollegeBasketballETL
from typing import List, Dict


class TeamsETL(BaseCollegeBasketballETL):
    """ETL for college basketball teams and rosters data"""
    
    def create_tables(self, cursor):
        """Create the teams and rosters tables if they don't exist"""
        
        # Teams table
        create_teams_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='teams' AND xtype='U')
        CREATE TABLE teams (
            id INT PRIMARY KEY,
            sourceId NVARCHAR(50),
            school NVARCHAR(200),
            mascot NVARCHAR(100),
            abbreviation NVARCHAR(20),
            displayName NVARCHAR(200),
            shortDisplayName NVARCHAR(200),
            primaryColor NVARCHAR(10),
            secondaryColor NVARCHAR(10),
            currentVenueId INT,
            currentVenue NVARCHAR(200),
            currentCity NVARCHAR(100),
            currentState NVARCHAR(10),
            conferenceId INT,
            conference NVARCHAR(50),
            
            INDEX IX_Teams_Conference (conferenceId),
            INDEX IX_Teams_School (school)
        )
        """
        
        # Rosters table
        create_rosters_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='rosters' AND xtype='U')
        CREATE TABLE rosters (
            id INT IDENTITY(1,1) PRIMARY KEY,
            teamId INT NOT NULL,
            teamSourceId NVARCHAR(50),
            team NVARCHAR(100),
            conference NVARCHAR(50),
            season INT NOT NULL,
            playerId INT NOT NULL,
            playerSourceId NVARCHAR(50),
            name NVARCHAR(200),
            firstName NVARCHAR(100),
            lastName NVARCHAR(100),
            jersey NVARCHAR(10),
            position NVARCHAR(50),
            height INT,
            weight INT,
            
            -- Hometown details
            hometownCountyFips NVARCHAR(10),
            hometownLongitude FLOAT,
            hometownLatitude FLOAT,
            hometownCountry NVARCHAR(100),
            hometownState NVARCHAR(50),
            hometownCity NVARCHAR(100),
            
            dateOfBirth DATE,
            startSeason INT,
            endSeason INT,
            
            CONSTRAINT UC_Roster UNIQUE (teamId, season, playerId),
            INDEX IX_Rosters_TeamId (teamId),
            INDEX IX_Rosters_Season (season),
            INDEX IX_Rosters_PlayerId (playerId),
            INDEX IX_Rosters_PlayerName (name)
        )
        """
        
        cursor.execute(create_teams_table_sql)
        cursor.execute(create_rosters_table_sql)
        print("Tables 'teams' and 'rosters' verified/created")
    
    def fetch_teams(self, season: int) -> List[Dict]:
        """
        Fetch teams for a specific season
        
        Args:
            season: Season year
            
        Returns:
            List of team dictionaries
        """
        endpoint = "/teams"
        params = {"season": season}
        return self.fetch_api(endpoint, params)
    
    def fetch_rosters(self, season: int) -> List[Dict]:
        """
        Fetch rosters for a specific season
        
        Args:
            season: Season year
            
        Returns:
            List of roster dictionaries (team + players array)
        """
        endpoint = "/teams/roster"
        params = {"season": season}
        return self.fetch_api(endpoint, params)
    
    def insert_teams(self, teams: List[Dict], cursor) -> int:
        """
        Insert teams into the database
        
        Args:
            teams: List of team dictionaries
            cursor: Database cursor
            
        Returns:
            Number of records inserted/updated
        """
        insert_sql = """
        MERGE teams AS target
        USING (SELECT ? AS id) AS source
        ON target.id = source.id
        WHEN MATCHED THEN
            UPDATE SET
                sourceId = ?,
                school = ?,
                mascot = ?,
                abbreviation = ?,
                displayName = ?,
                shortDisplayName = ?,
                primaryColor = ?,
                secondaryColor = ?,
                currentVenueId = ?,
                currentVenue = ?,
                currentCity = ?,
                currentState = ?,
                conferenceId = ?,
                conference = ?
        WHEN NOT MATCHED THEN
            INSERT (id, sourceId, school, mascot, abbreviation, displayName, 
                    shortDisplayName, primaryColor, secondaryColor, currentVenueId,
                    currentVenue, currentCity, currentState, conferenceId, conference)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for team in teams:
            try:
                params = (
                    # For MERGE condition
                    team.get('id'),
                    # For UPDATE
                    team.get('sourceId'),
                    team.get('school'),
                    team.get('mascot'),
                    team.get('abbreviation'),
                    team.get('displayName'),
                    team.get('shortDisplayName'),
                    team.get('primaryColor'),
                    team.get('secondaryColor'),
                    team.get('currentVenueId'),
                    team.get('currentVenue'),
                    team.get('currentCity'),
                    team.get('currentState'),
                    team.get('conferenceId'),
                    team.get('conference'),
                    # For INSERT
                    team.get('id'),
                    team.get('sourceId'),
                    team.get('school'),
                    team.get('mascot'),
                    team.get('abbreviation'),
                    team.get('displayName'),
                    team.get('shortDisplayName'),
                    team.get('primaryColor'),
                    team.get('secondaryColor'),
                    team.get('currentVenueId'),
                    team.get('currentVenue'),
                    team.get('currentCity'),
                    team.get('currentState'),
                    team.get('conferenceId'),
                    team.get('conference')
                )
                
                if self.execute_merge(cursor, insert_sql, params, 
                                     f"team {team.get('school')}"):
                    inserted += 1
                    
            except Exception as e:
                print(f"Error processing team {team.get('school')}: {e}")
                continue
        
        return inserted
    
    def insert_rosters(self, rosters_data: List[Dict], cursor) -> int:
        """
        Insert rosters into the database
        
        Args:
            rosters_data: List of roster dictionaries (each contains team info + players array)
            cursor: Database cursor
            
        Returns:
            Number of player records inserted
        """
        insert_sql = """
        MERGE rosters AS target
        USING (SELECT ? AS teamId, ? AS season, ? AS playerId) AS source
        ON target.teamId = source.teamId 
           AND target.season = source.season 
           AND target.playerId = source.playerId
        WHEN NOT MATCHED THEN
            INSERT (teamId, teamSourceId, team, conference, season, 
                    playerId, playerSourceId, name, firstName, lastName, 
                    jersey, position, height, weight,
                    hometownCountyFips, hometownLongitude, hometownLatitude,
                    hometownCountry, hometownState, hometownCity,
                    dateOfBirth, startSeason, endSeason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for roster in rosters_data:
            team_id = roster.get('teamId')
            team_source_id = roster.get('teamSourceId')
            team_name = roster.get('team')
            conference = roster.get('conference')
            season = roster.get('season')
            
            players = roster.get('players', [])
            
            for player in players:
                try:
                    hometown = player.get('hometown', {})
                    
                    # Parse date of birth
                    dob = player.get('dateOfBirth')
                    if dob:
                        try:
                            from datetime import datetime
                            dob = datetime.strptime(dob, '%Y-%m-%d').date()
                        except:
                            dob = None
                    
                    params = (
                        # For MERGE condition
                        team_id,
                        season,
                        player.get('id'),
                        # For INSERT
                        team_id,
                        team_source_id,
                        team_name,
                        conference,
                        season,
                        player.get('id'),
                        player.get('sourceId'),
                        player.get('name'),
                        player.get('firstName'),
                        player.get('lastName'),
                        player.get('jersey'),
                        player.get('position'),
                        player.get('height'),
                        player.get('weight'),
                        hometown.get('countyFips'),
                        hometown.get('longitude'),
                        hometown.get('latitude'),
                        hometown.get('country'),
                        hometown.get('state'),
                        hometown.get('city'),
                        dob,
                        player.get('startSeason'),
                        player.get('endSeason')
                    )
                    
                    if self.execute_merge(cursor, insert_sql, params, 
                                         f"player {player.get('name')} on team {team_name}"):
                        inserted += 1
                        
                except Exception as e:
                    print(f"Error processing player {player.get('name')}: {e}")
                    continue
        
        return inserted
    
    def run_etl(self, start_season: int = 2006, end_season: int = 2025,
                fetch_teams: bool = True, fetch_rosters: bool = True):
        """
        Run the complete ETL process for teams and rosters
        
        Args:
            start_season: Starting season year
            end_season: Ending season year
            fetch_teams: Whether to fetch teams data (default True)
            fetch_rosters: Whether to fetch rosters data (default True)
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Create tables if needed
            self.create_tables(cursor)
            conn.commit()
            
            total_teams_inserted = 0
            total_rosters_inserted = 0
            
            # Process each season
            for season in range(start_season, end_season + 1):
                print(f"\n{'='*60}")
                print(f"Processing season {season}...")
                print(f"{'='*60}")
                
                # Fetch and insert teams
                if fetch_teams:
                    print(f"\n--- Teams Endpoint ---")
                    teams = self.fetch_teams(season)
                    
                    if teams:
                        inserted = self.insert_teams(teams, cursor)
                        conn.commit()
                        total_teams_inserted += inserted
                        print(f"✓ Inserted/updated {inserted} teams for season {season}")
                    
                    self.rate_limit_sleep(0.5)
                
                # Fetch and insert rosters
                if fetch_rosters:
                    print(f"\n--- Rosters Endpoint ---")
                    rosters = self.fetch_rosters(season)
                    
                    if rosters:
                        inserted = self.insert_rosters(rosters, cursor)
                        conn.commit()
                        total_rosters_inserted += inserted
                        print(f"✓ Inserted {inserted} player records for season {season}")
                    
                    self.rate_limit_sleep(0.5)
            
            print(f"\n{'='*60}")
            print(f"=== ETL Complete ===")
            print(f"{'='*60}")
            if fetch_teams:
                print(f"Total teams inserted/updated: {total_teams_inserted}")
            if fetch_rosters:
                print(f"Total roster records inserted: {total_rosters_inserted}")
            
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
    etl = TeamsETL(API_KEY, DB_CONNECTION)
    
    # Fetch both teams and rosters (default)
    etl.run_etl(start_season=2006, end_season=2025)
    
    # Or fetch only teams
    # etl.run_etl(start_season=2006, end_season=2025, fetch_teams=True, fetch_rosters=False)
    
    # Or fetch only rosters
    # etl.run_etl(start_season=2006, end_season=2025, fetch_teams=False, fetch_rosters=True)