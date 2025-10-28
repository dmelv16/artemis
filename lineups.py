from base_etl import BaseCollegeBasketballETL, GameIDFetcher
from typing import List, Dict


class LineupsETL(BaseCollegeBasketballETL):
    """ETL for college basketball lineups (5-man units) data"""
    
    def create_table(self, cursor):
        """Create the lineups table if it doesn't exist"""
        create_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='lineups' AND xtype='U')
        CREATE TABLE lineups (
            id INT IDENTITY(1,1) PRIMARY KEY,
            gameId INT NOT NULL,
            teamId INT NOT NULL,
            team NVARCHAR(100),
            conference NVARCHAR(50),
            idHash NVARCHAR(100),
            athletes NVARCHAR(MAX),
            totalSeconds INT,
            pace FLOAT,
            offenseRating FLOAT,
            defenseRating FLOAT,
            netRating FLOAT,
            
            -- Team Stats
            points INT,
            possessions INT,
            assists INT,
            steals INT,
            turnovers INT,
            blocks INT,
            defensiveRebounds INT,
            offensiveRebounds INT,
            trueShooting FLOAT,
            
            -- Field Goals
            fgMade INT,
            fgAttempted INT,
            fgPct FLOAT,
            ftMade INT,
            ftAttempted INT,
            ftPct FLOAT,
            
            -- Two Pointers
            twoPtMade INT,
            twoPtAttempted INT,
            twoPtPct FLOAT,
            tipInsMade INT,
            tipInsAttempted INT,
            tipInsPct FLOAT,
            dunksMade INT,
            dunksAttempted INT,
            dunksPct FLOAT,
            layupsMade INT,
            layupsAttempted INT,
            layupsPct FLOAT,
            jumpersMade INT,
            jumpersAttempted INT,
            jumpersPct FLOAT,
            
            -- Three Pointers
            threePtMade INT,
            threePtAttempted INT,
            threePtPct FLOAT,
            
            -- Four Factors
            effectiveFieldGoalPct FLOAT,
            turnoverRatio FLOAT,
            offensiveReboundPct FLOAT,
            freeThrowRate FLOAT,
            
            -- Opponent Stats
            oppPoints INT,
            oppPossessions INT,
            oppAssists INT,
            oppSteals INT,
            oppTurnovers INT,
            oppBlocks INT,
            oppDefensiveRebounds INT,
            oppOffensiveRebounds INT,
            oppTrueShooting FLOAT,
            
            -- Opponent Field Goals
            oppFgMade INT,
            oppFgAttempted INT,
            oppFgPct FLOAT,
            oppFtMade INT,
            oppFtAttempted INT,
            oppFtPct FLOAT,
            
            -- Opponent Two Pointers
            oppTwoPtMade INT,
            oppTwoPtAttempted INT,
            oppTwoPtPct FLOAT,
            oppTipInsMade INT,
            oppTipInsAttempted INT,
            oppTipInsPct FLOAT,
            oppDunksMade INT,
            oppDunksAttempted INT,
            oppDunksPct FLOAT,
            oppLayupsMade INT,
            oppLayupsAttempted INT,
            oppLayupsPct FLOAT,
            oppJumpersMade INT,
            oppJumpersAttempted INT,
            oppJumpersPct FLOAT,
            
            -- Opponent Three Pointers
            oppThreePtMade INT,
            oppThreePtAttempted INT,
            oppThreePtPct FLOAT,
            
            -- Opponent Four Factors
            oppEffectiveFieldGoalPct FLOAT,
            oppTurnoverRatio FLOAT,
            oppOffensiveReboundPct FLOAT,
            oppFreeThrowRate FLOAT,
            
            CONSTRAINT UC_Lineup UNIQUE (gameId, teamId, idHash),
            INDEX IX_Lineups_GameId (gameId),
            INDEX IX_Lineups_TeamId (teamId),
            INDEX IX_Lineups_IdHash (idHash)
        )
        """
        cursor.execute(create_table_sql)
        print("Table 'lineups' verified/created")
    
    def fetch_lineups_for_game(self, game_id: int) -> List[Dict]:
        """
        Fetch lineups for a specific game
        
        Args:
            game_id: The game ID
            
        Returns:
            List of lineup dictionaries
        """
        endpoint = f"/lineups/game/{game_id}"
        return self.fetch_api(endpoint)
    
    def insert_lineups(self, lineups: List[Dict], cursor) -> int:
        """
        Insert lineups into the database
        
        Args:
            lineups: List of lineup dictionaries
            cursor: Database cursor
            
        Returns:
            Number of records inserted
        """
        insert_sql = """
        MERGE lineups AS target
        USING (SELECT ? AS gameId, ? AS teamId, ? AS idHash) AS source
        ON target.gameId = source.gameId 
           AND target.teamId = source.teamId 
           AND target.idHash = source.idHash
        WHEN NOT MATCHED THEN
            INSERT (gameId, teamId, team, conference, idHash, athletes, totalSeconds, 
                    pace, offenseRating, defenseRating, netRating,
                    points, possessions, assists, steals, turnovers, blocks, 
                    defensiveRebounds, offensiveRebounds, trueShooting,
                    fgMade, fgAttempted, fgPct, ftMade, ftAttempted, ftPct,
                    twoPtMade, twoPtAttempted, twoPtPct, tipInsMade, tipInsAttempted, tipInsPct,
                    dunksMade, dunksAttempted, dunksPct, layupsMade, layupsAttempted, layupsPct,
                    jumpersMade, jumpersAttempted, jumpersPct, threePtMade, threePtAttempted, threePtPct,
                    effectiveFieldGoalPct, turnoverRatio, offensiveReboundPct, freeThrowRate,
                    oppPoints, oppPossessions, oppAssists, oppSteals, oppTurnovers, oppBlocks,
                    oppDefensiveRebounds, oppOffensiveRebounds, oppTrueShooting,
                    oppFgMade, oppFgAttempted, oppFgPct, oppFtMade, oppFtAttempted, oppFtPct,
                    oppTwoPtMade, oppTwoPtAttempted, oppTwoPtPct, oppTipInsMade, oppTipInsAttempted, oppTipInsPct,
                    oppDunksMade, oppDunksAttempted, oppDunksPct, oppLayupsMade, oppLayupsAttempted, oppLayupsPct,
                    oppJumpersMade, oppJumpersAttempted, oppJumpersPct, oppThreePtMade, oppThreePtAttempted, oppThreePtPct,
                    oppEffectiveFieldGoalPct, oppTurnoverRatio, oppOffensiveReboundPct, oppFreeThrowRate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for lineup in lineups:
            try:
                team_stats = lineup.get('teamStats', {})
                opp_stats = lineup.get('opponentStats', {})
                
                # Team shooting
                team_fg = team_stats.get('fieldGoals', {})
                team_ft = team_stats.get('freeThrows', {})
                team_2pt = team_stats.get('twoPointers', {})
                team_tip = team_2pt.get('tipIns', {}) if team_2pt else {}
                team_dunk = team_2pt.get('dunks', {}) if team_2pt else {}
                team_layup = team_2pt.get('layups', {}) if team_2pt else {}
                team_jump = team_2pt.get('jumpers', {}) if team_2pt else {}
                team_3pt = team_stats.get('threePointers', {})
                team_ff = team_stats.get('fourFactors', {})
                
                # Opponent shooting
                opp_fg = opp_stats.get('fieldGoals', {})
                opp_ft = opp_stats.get('freeThrows', {})
                opp_2pt = opp_stats.get('twoPointers', {})
                opp_tip = opp_2pt.get('tipIns', {}) if opp_2pt else {}
                opp_dunk = opp_2pt.get('dunks', {}) if opp_2pt else {}
                opp_layup = opp_2pt.get('layups', {}) if opp_2pt else {}
                opp_jump = opp_2pt.get('jumpers', {}) if opp_2pt else {}
                opp_3pt = opp_stats.get('threePointers', {})
                opp_ff = opp_stats.get('fourFactors', {})
                
                params = (
                    # For MERGE condition
                    lineup.get('gameId'),
                    lineup.get('teamId'),
                    lineup.get('idHash'),
                    # For INSERT - game/lineup info
                    lineup.get('gameId'),
                    lineup.get('teamId'),
                    lineup.get('team'),
                    lineup.get('conference'),
                    lineup.get('idHash'),
                    self.json_serialize(lineup.get('athletes')),
                    lineup.get('totalSeconds'),
                    lineup.get('pace'),
                    lineup.get('offenseRating'),
                    lineup.get('defenseRating'),
                    lineup.get('netRating'),
                    # Team stats
                    team_stats.get('points'),
                    team_stats.get('possessions'),
                    team_stats.get('assists'),
                    team_stats.get('steals'),
                    team_stats.get('turnovers'),
                    team_stats.get('blocks'),
                    team_stats.get('defensiveRebounds'),
                    team_stats.get('offensiveRebounds'),
                    team_stats.get('trueShooting'),
                    # Team shooting
                    team_fg.get('made'),
                    team_fg.get('attempted'),
                    team_fg.get('pct'),
                    team_ft.get('made'),
                    team_ft.get('attempted'),
                    team_ft.get('pct'),
                    team_2pt.get('made'),
                    team_2pt.get('attempted'),
                    team_2pt.get('pct'),
                    team_tip.get('made'),
                    team_tip.get('attempted'),
                    team_tip.get('pct'),
                    team_dunk.get('made'),
                    team_dunk.get('attempted'),
                    team_dunk.get('pct'),
                    team_layup.get('made'),
                    team_layup.get('attempted'),
                    team_layup.get('pct'),
                    team_jump.get('made'),
                    team_jump.get('attempted'),
                    team_jump.get('pct'),
                    team_3pt.get('made'),
                    team_3pt.get('attempted'),
                    team_3pt.get('pct'),
                    # Team four factors
                    team_ff.get('effectiveFieldGoalPct'),
                    team_ff.get('turnoverRatio'),
                    team_ff.get('offensiveReboundPct'),
                    team_ff.get('freeThrowRate'),
                    # Opponent stats
                    opp_stats.get('points'),
                    opp_stats.get('possessions'),
                    opp_stats.get('assists'),
                    opp_stats.get('steals'),
                    opp_stats.get('turnovers'),
                    opp_stats.get('blocks'),
                    opp_stats.get('defensiveRebounds'),
                    opp_stats.get('offensiveRebounds'),
                    opp_stats.get('trueShooting'),
                    # Opponent shooting
                    opp_fg.get('made'),
                    opp_fg.get('attempted'),
                    opp_fg.get('pct'),
                    opp_ft.get('made'),
                    opp_ft.get('attempted'),
                    opp_ft.get('pct'),
                    opp_2pt.get('made'),
                    opp_2pt.get('attempted'),
                    opp_2pt.get('pct'),
                    opp_tip.get('made'),
                    opp_tip.get('attempted'),
                    opp_tip.get('pct'),
                    opp_dunk.get('made'),
                    opp_dunk.get('attempted'),
                    opp_dunk.get('pct'),
                    opp_layup.get('made'),
                    opp_layup.get('attempted'),
                    opp_layup.get('pct'),
                    opp_jump.get('made'),
                    opp_jump.get('attempted'),
                    opp_jump.get('pct'),
                    opp_3pt.get('made'),
                    opp_3pt.get('attempted'),
                    opp_3pt.get('pct'),
                    # Opponent four factors
                    opp_ff.get('effectiveFieldGoalPct'),
                    opp_ff.get('turnoverRatio'),
                    opp_ff.get('offensiveReboundPct'),
                    opp_ff.get('freeThrowRate')
                )
                
                if self.execute_merge(cursor, insert_sql, params, 
                                     f"game {lineup.get('gameId')} lineup {lineup.get('idHash')}"):
                    inserted += 1
                    
            except Exception as e:
                print(f"Error processing lineup: {e}")
                continue
        
        return inserted
    
    def process_game(self, game_id: int, cursor) -> int:
        """
        Process lineups for a single game
        
        Args:
            game_id: Game ID to process
            cursor: Database cursor
            
        Returns:
            Number of lineups inserted
        """
        lineups = self.fetch_lineups_for_game(game_id)
        
        if not lineups:
            return 0
        
        inserted = self.insert_lineups(lineups, cursor)
        return inserted
    
    def run_etl(self, start_season: int = 2006, end_season: int = 2025, 
                batch_size: int = 100):
        """
        Run the complete ETL process for lineups
        
        Args:
            start_season: Starting season year
            end_season: Ending season year
            batch_size: Number of games to process before committing
        """
        # First, get all game IDs from the database
        fetcher = GameIDFetcher(self.conn_string)
        all_game_ids = fetcher.get_all_game_ids(start_season, end_season)
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Create table if needed
            self.create_table(cursor)
            conn.commit()
            
            total_inserted = 0
            
            # Process each season
            for season in range(start_season, end_season + 1):
                game_ids = all_game_ids.get(season, [])
                
                if not game_ids:
                    print(f"No games found for season {season}")
                    continue
                
                print(f"\n{'='*60}")
                print(f"Processing {len(game_ids)} games for season {season}...")
                print(f"{'='*60}")
                
                season_inserted = 0
                
                # Process games in batches
                for i, game_id in enumerate(game_ids, 1):
                    try:
                        inserted = self.process_game(game_id, cursor)
                        season_inserted += inserted
                        
                        # Commit in batches
                        if i % batch_size == 0:
                            conn.commit()
                            print(f"  Processed {i}/{len(game_ids)} games... ({season_inserted} lineups)")
                        
                        # Rate limiting
                        self.rate_limit_sleep(0.3)
                        
                    except Exception as e:
                        print(f"Error processing game {game_id}: {e}")
                        continue
                
                # Final commit for the season
                conn.commit()
                total_inserted += season_inserted
                print(f"✓ Season {season} complete: {season_inserted} lineups inserted")
            
            print(f"\n{'='*60}")
            print(f"=== ETL Complete ===")
            print(f"{'='*60}")
            print(f"Total lineups inserted: {total_inserted}")
            
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
    etl = LineupsETL(API_KEY, DB_CONNECTION)
    
    # Process all seasons
    etl.run_etl(start_season=2006, end_season=2025, batch_size=100)
    
    # Or process a single season
    # etl.run_etl(start_season=2024, end_season=2024, batch_size=50)