"""
API Football Module for Head-to-Head Data
Handles H2H data retrieval with fallback mechanisms
"""

import os
import requests
import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class APIFootballService:
    def __init__(self):
        self.api_key = os.environ.get('APIFOOTBALL_API_KEY', '908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485')
        self.base_url = "https://apiv3.apifootball.com/"
        self.cache = {}
        
    def get_head_to_head_data(self, home_team_id, away_team_id, limit=10):
        """
        Get head-to-head data between two teams
        
        Args:
            home_team_id: ID of home team
            away_team_id: ID of away team
            limit: Maximum number of matches to return
            
        Returns:
            list: List of H2H matches or empty list if none found
        """
        try:
            # Create cache key
            cache_key = f"h2h_{home_team_id}_{away_team_id}"
            
            # Check cache first
            if cache_key in self.cache:
                logger.debug(f"H2H data found in cache for {home_team_id} vs {away_team_id}")
                return self.cache[cache_key]
            
            # Prepare API request
            params = {
                'action': 'get_H2H',
                'firstTeam': home_team_id,
                'secondTeam': away_team_id,
                'APIkey': self.api_key
            }
            
            logger.info(f"Fetching H2H data: {home_team_id} vs {away_team_id}")
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list) and len(data) > 0:
                    # Process and format the data
                    h2h_matches = self._process_h2h_data(data[:limit])
                    
                    # Cache the result
                    self.cache[cache_key] = h2h_matches
                    
                    logger.info(f"H2H data retrieved successfully: {len(h2h_matches)} matches")
                    return h2h_matches
                else:
                    logger.warning(f"No H2H data found for {home_team_id} vs {away_team_id}")
                    return []
            else:
                logger.error(f"API request failed: {response.status_code}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching H2H data: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error fetching H2H data: {str(e)}")
            return []
    
    def _process_h2h_data(self, raw_data):
        """Process raw H2H data from API"""
        processed_matches = []
        
        for match in raw_data:
            try:
                processed_match = {
                    'match_id': match.get('match_id', ''),
                    'match_date': match.get('match_date', ''),
                    'home_team': {
                        'id': match.get('match_hometeam_id', ''),
                        'name': match.get('match_hometeam_name', ''),
                        'score': int(match.get('match_hometeam_score', 0))
                    },
                    'away_team': {
                        'id': match.get('match_awayteam_id', ''),
                        'name': match.get('match_awayteam_name', ''),
                        'score': int(match.get('match_awayteam_score', 0))
                    },
                    'status': match.get('match_status', ''),
                    'league': match.get('league_name', '')
                }
                
                # Add derived data
                processed_match['total_goals'] = processed_match['home_team']['score'] + processed_match['away_team']['score']
                processed_match['both_teams_scored'] = processed_match['home_team']['score'] > 0 and processed_match['away_team']['score'] > 0
                
                if processed_match['home_team']['score'] > processed_match['away_team']['score']:
                    processed_match['result'] = 'HOME_WIN'
                elif processed_match['home_team']['score'] < processed_match['away_team']['score']:
                    processed_match['result'] = 'AWAY_WIN'
                else:
                    processed_match['result'] = 'DRAW'
                
                processed_matches.append(processed_match)
                
            except Exception as e:
                logger.warning(f"Error processing H2H match data: {str(e)}")
                continue
        
        return processed_matches

# Global service instance
api_football_service = APIFootballService()

def get_head_to_head_data(home_team_id, away_team_id, limit=10):
    """
    Get head-to-head data between two teams
    Main function for external use
    """
    return api_football_service.get_head_to_head_data(home_team_id, away_team_id, limit)

def get_h2h_statistics(home_team_id, away_team_id):
    """
    Get H2H statistics summary
    """
    try:
        h2h_data = get_head_to_head_data(home_team_id, away_team_id)
        
        if not h2h_data:
            return None
        
        stats = {
            'total_matches': len(h2h_data),
            'home_wins': 0,
            'away_wins': 0,
            'draws': 0,
            'total_goals': 0,
            'avg_goals_per_match': 0,
            'btts_count': 0,
            'btts_percentage': 0,
            'over_2_5_count': 0,
            'over_2_5_percentage': 0,
            'last_5_results': []
        }
        
        for match in h2h_data:
            # Count results
            if match['result'] == 'HOME_WIN':
                stats['home_wins'] += 1
            elif match['result'] == 'AWAY_WIN':
                stats['away_wins'] += 1
            else:
                stats['draws'] += 1
            
            # Count goals
            stats['total_goals'] += match['total_goals']
            
            # Count BTTS
            if match['both_teams_scored']:
                stats['btts_count'] += 1
            
            # Count Over 2.5
            if match['total_goals'] > 2:
                stats['over_2_5_count'] += 1
        
        # Calculate percentages
        total = stats['total_matches']
        stats['avg_goals_per_match'] = round(stats['total_goals'] / total, 2) if total > 0 else 0
        stats['btts_percentage'] = round((stats['btts_count'] / total) * 100, 1) if total > 0 else 0
        stats['over_2_5_percentage'] = round((stats['over_2_5_count'] / total) * 100, 1) if total > 0 else 0
        
        # Last 5 results
        stats['last_5_results'] = [match['result'] for match in h2h_data[:5]]
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating H2H statistics: {str(e)}")
        return None