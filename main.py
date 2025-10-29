"""
Main ETL Orchestrator for College Basketball Data
Runs all ETL processes in the correct order
"""

from datetime import datetime

# Import all ETL classes
from conferences import ConferencesETL
from venues import VenuesETL
from teams import TeamsETL
from games import GamesETL
from rankings import RankingsETL
from lines import LinesETL
from recruiting import RecruitingETL
from substitution import SubstitutionsETL
from plays import PlaysETL
from lineups import LineupsETL


# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================
API_KEY = "ffWPBYbZcxYH+eWEuwu3LVDuAdaRD/1tvzaIe2FkAQ5uj+V4UNvyaVyNW/O2Sx4B"
DB_SERVER = "DESKTOP-J9IV3OH"
DB_NAME = "cbbDB"
DB_USER = None  # Set to None for Windows Authentication
DB_PASSWORD = None  # Set to None for Windows Authentication

START_SEASON = 2006
END_SEASON = 2025
INCLUDE_GAME_DETAILS = True  # Set to True to include substitutions, plays, lineups (very slow!)
BATCH_SIZE = 100

# Select which phases to run (comment out phases you don't want)
PHASES_TO_RUN = [
    'reference',      # Conferences & Venues
    'season',         # Teams, Games, Rankings, Lines
    'recruiting',     # Recruiting data
    'game-details', # Substitutions, Plays, Lineups (VERY SLOW - uncomment to enable)
]
# ============================================================================


class MainETL:
    """Main orchestrator for all college basketball ETL processes"""
    
    def __init__(self, api_key: str, db_connection: str):
        self.api_key = api_key
        self.db_connection = db_connection
        self.start_time = None
        
    def log(self, message: str):
        """Print timestamped log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def run_reference_data(self):
        """Run one-time reference data ETL (conferences, venues)"""
        self.log("="*80)
        self.log("PHASE 1: Reference Data (Conferences & Venues)")
        self.log("="*80)
        
        try:
            # Conferences
            # self.log("Running Conferences ETL...")
            # conferences_etl = ConferencesETL(self.api_key, self.db_connection)
            # conferences_etl.run_etl()
            # self.log("✓ Conferences complete\n")
            
            # # Venues
            # self.log("Running Venues ETL...")
            # venues_etl = VenuesETL(self.api_key, self.db_connection)
            # venues_etl.run_etl()
            self.log("✓ Venues complete\n")
            
        except Exception as e:
            self.log(f"✗ Error in reference data: {e}")
            raise
    
    def run_season_data(self, start_season: int, end_season: int):
        """Run season-based ETL (teams, games, rankings, etc.)"""
        self.log("="*80)
        self.log(f"PHASE 2: Season Data ({start_season}-{end_season})")
        self.log("="*80)
        
        try:
            # Teams & Rosters
            # self.log("Running Teams & Rosters ETL...")
            # teams_etl = TeamsETL(self.api_key, self.db_connection)
            # teams_etl.run_etl(start_season, end_season)
            # self.log("✓ Teams & Rosters complete\n")
            
            # Games & Team Games
            self.log("Running Games & Team Games ETL...")
            games_etl = GamesETL(self.api_key, self.db_connection)
            games_etl.run_etl(start_season, end_season)
            self.log("✓ Games & Team Games complete\n")
            
            # Rankings
            self.log("Running Rankings ETL...")
            rankings_etl = RankingsETL(self.api_key, self.db_connection)
            rankings_etl.run_etl(start_season, end_season)
            self.log("✓ Rankings complete\n")
            
            # Betting Lines
            self.log("Running Betting Lines ETL...")
            lines_etl = LinesETL(self.api_key, self.db_connection)
            lines_etl.run_etl(start_season, end_season)
            self.log("✓ Betting Lines complete\n")
            
        except Exception as e:
            self.log(f"✗ Error in season data: {e}")
            raise
    
    def run_recruiting_data(self, start_year: int, end_year: int):
        """Run recruiting data ETL"""
        self.log("="*80)
        self.log(f"PHASE 3: Recruiting Data ({start_year}-{end_year})")
        self.log("="*80)
        
        try:
            self.log("Running Recruiting ETL...")
            recruiting_etl = RecruitingETL(self.api_key, self.db_connection)
            recruiting_etl.run_etl(start_year, end_year)
            self.log("✓ Recruiting complete\n")
            
        except Exception as e:
            self.log(f"✗ Error in recruiting data: {e}")
            raise
    
    def run_game_detail_data(self, start_season: int, end_season: int, 
                            batch_size: int = 100):
        """Run game-level detail ETL (substitutions, plays, lineups)"""
        self.log("="*80)
        self.log(f"PHASE 4: Game Detail Data ({start_season}-{end_season})")
        self.log("="*80)
        self.log("WARNING: This phase is time-intensive (requires fetching each game individually)")
        
        try:
            # Substitutions
            self.log("Running Substitutions ETL...")
            substitutions_etl = SubstitutionsETL(self.api_key, self.db_connection)
            substitutions_etl.run_etl(start_season, end_season, batch_size)
            self.log("✓ Substitutions complete\n")
            
            # Play-by-Play
            self.log("Running Play-by-Play ETL...")
            plays_etl = PlaysETL(self.api_key, self.db_connection)
            plays_etl.run_etl(start_season, end_season, batch_size)
            self.log("✓ Play-by-Play complete\n")
            
            # Lineups
            self.log("Running Lineups ETL...")
            lineups_etl = LineupsETL(self.api_key, self.db_connection)
            lineups_etl.run_etl(start_season, end_season, batch_size)
            self.log("✓ Lineups complete\n")
            
        except Exception as e:
            self.log(f"✗ Error in game detail data: {e}")
            raise
    
    def run_etl_pipeline(self, phases: list, start_season: int, end_season: int,
                        batch_size: int = 100):
        """Run ETL pipeline with specified phases"""
        self.start_time = datetime.now()
        self.log("\n" + "="*80)
        self.log("STARTING ETL PIPELINE")
        self.log("="*80 + "\n")
        
        try:
            # Phase 1: Reference Data
            if 'reference' in phases:
                self.run_reference_data()
            
            # Phase 2: Season Data
            if 'season' in phases:
                self.run_season_data(start_season, end_season)
            
            # Phase 3: Recruiting
            if 'recruiting' in phases:
                self.run_recruiting_data(start_season, end_season)
            
            # Phase 4: Game Details
            if 'game-details' in phases:
                self.run_game_detail_data(start_season, end_season, batch_size)
            
            # Complete
            elapsed = datetime.now() - self.start_time
            self.log("="*80)
            self.log("ETL PIPELINE COMPLETE!")
            self.log("="*80)
            self.log(f"Total time: {elapsed}")
            self.log("")
            
        except Exception as e:
            elapsed = datetime.now() - self.start_time
            self.log("="*80)
            self.log(f"ETL PIPELINE FAILED: {e}")
            self.log("="*80)
            self.log(f"Time elapsed: {elapsed}")
            raise


def main():
    """Main entry point"""
    
    # Build connection string
    if DB_USER and DB_PASSWORD:
        db_connection = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_NAME};"
            f"UID={DB_USER};"
            f"PWD={DB_PASSWORD}"
        )
    else:
        # Windows Authentication
        db_connection = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_NAME};"
            f"Trusted_Connection=yes;"
        )
    
    # Initialize and run ETL
    main_etl = MainETL(API_KEY, db_connection)
    main_etl.run_etl_pipeline(
        phases=PHASES_TO_RUN,
        start_season=START_SEASON,
        end_season=END_SEASON,
        batch_size=BATCH_SIZE
    )


if __name__ == "__main__":
    main()