import psycopg2 
from psycopg2.extras import RealDictCursor, Json 
from datetime import datetime 
from typing import Optional, List, Dict, Any 
import uuid 
import json 
 
class ChatHistoryManager: 
    """ 
    Manages chat history for RAG chatbot using PostgreSQL. 
    Handles sessions, messages, and conversation context. 
    """ 
     
    def __init__(self, db_config: Dict[str, str]): 
        """ 
        Initialize the chat history manager. 
         
        Args: 
            db_config: Dictionary with keys: host, database, user, password, 
port 
        """ 
        self.db_config = db_config 
        self.connection = None 
     
    def _get_connection(self): 
        """Get or create database connection.""" 

        if self.connection is None or self.connection.closed: 
            self.connection = psycopg2.connect( 
                host=self.db_config['host'], 
                database=self.db_config['database'], 
                user=self.db_config['user'], 
                password=self.db_config['password'], 
                port=self.db_config.get('port', 5432) 
            ) 
        return self.connection 
     
    def create_session(self, user_id: str) -> str: 
        """ 
        Create a new chat session for a user. 
         
        Args: 
            user_id: Unique identifier for the user 
             
        Returns: 
            session_id: UUID of the created session 
        """ 
        conn = self._get_connection() 
        cursor = conn.cursor() 
         
        try: 
            cursor.execute( 
                """ 
                INSERT INTO chat.sessions (user_id) 
                VALUES (%s) 
                RETURNING id 
                """, 
                (user_id,) 
            ) 
            session_id = cursor.fetchone()[0] 
            conn.commit() 
            return str(session_id) 
        except Exception as e: 
            conn.rollback() 
            raise Exception(f"Error creating session: {str(e)}") 
        finally: 
            cursor.close() 
     
    def get_or_create_session(self, user_id: str, session_id: Optional[str] = 
None) -> str: 
        """ 
        Get existing session or create new one. 
         
        Args: 
            user_id: Unique identifier for the user 

            session_id: Optional existing session ID 
             
        Returns: 
            session_id: UUID of the session 
        """ 
        if session_id: 
            # Verify session exists and belongs to user 
            if self.verify_session(user_id, session_id): 
                return session_id 
         
        # Create new session 
        return self.create_session(user_id) 
     
    def verify_session(self, user_id: str, session_id: str) -> bool: 
        """ 
        Verify that a session exists and belongs to the user. 
         
        Args: 
            user_id: Unique identifier for the user 
            session_id: Session UUID to verify 
             
        Returns: 
            bool: True if session is valid, False otherwise 
        """ 
        conn = self._get_connection() 
        cursor = conn.cursor() 
         
        try: 
            cursor.execute( 
                """ 
                SELECT EXISTS( 
                    SELECT 1 FROM chat.sessions 
                    WHERE id = %s AND user_id = %s 
                ) 
                """, 
                (session_id, user_id) 
            ) 
            return cursor.fetchone()[0] 
        finally: 
            cursor.close() 
     
    def store_user_history( 
        self, 
        user_id: str, 
        session_id: str, 
        query: str, 
        response: str, 
        metadata: Optional[Dict[str, Any]] = None 

    ) -> tuple: 
        """ 
        Store a user query and assistant response in the chat history. 
         
        Args: 
            user_id: Unique identifier for the user 
            session_id: Session UUID 
            query: User's question/input 
            response: Assistant's response 
            metadata: Optional metadata (model info, tokens, etc.) 
             
        Returns: 
            tuple: (query_message_id, response_message_id) 
        """ 
        conn = self._get_connection() 
        cursor = conn.cursor() 
         
        # Verify session 
        if not self.verify_session(user_id, session_id): 
            raise ValueError(f"Invalid session_id {session_id} for user  {user_id}") 
         
        metadata = metadata or {} 
        metadata['timestamp'] = datetime.utcnow().isoformat() 
         
        try: 
            # Store user query 
            cursor.execute( 
                """ 
                INSERT INTO chat.messages (session_id, role, content, 
metadata) 
                VALUES (%s, %s, %s, %s) 
                RETURNING id 
                """, 
                (session_id, 'user', query, Json(metadata)) 
            ) 
            query_message_id = cursor.fetchone()[0] 
             
            # Store assistant response 
            cursor.execute( 
                """ 
                INSERT INTO chat.messages (session_id, role, content, 
metadata) 
                VALUES (%s, %s, %s, %s) 
                RETURNING id 
                """, 
                (session_id, 'assistant', response, Json(metadata)) 
            ) 

            response_message_id = cursor.fetchone()[0] 
             
            conn.commit() 
            return (str(query_message_id), str(response_message_id)) 
         
        except Exception as e: 
            conn.rollback() 
            raise Exception(f"Error storing chat history: {str(e)}") 
        finally: 
            cursor.close() 
     
    def get_user_history( 
        self, 
        user_id: str, 
        session_id: str, 
        limit: Optional[int] = None 
    ) -> Optional[Dict[str, Any]]: 
        """ 
        Retrieve chat history for a specific session. 
         
        Args: 
            user_id: Unique identifier for the user 
            session_id: Session UUID 
            limit: Optional limit on number of messages to retrieve 
             
        Returns: 
            Dictionary containing session info and interactions, or None if 
not found 
        """ 
        conn = self._get_connection() 
        cursor = conn.cursor(cursor_factory=RealDictCursor) 
         
        # Verify session 
        if not self.verify_session(user_id, session_id): 
            return None 
         
        try: 
            # Get session info 
            cursor.execute( 
                """ 
                SELECT id, user_id, created_at 
                FROM chat.sessions 
                WHERE id = %s AND user_id = %s 
                """, 
                (session_id, user_id) 
            ) 
            session_info = cursor.fetchone() 
             

            if not session_info: 
                return None 
             
            # Get messages 
            query = """ 
                SELECT id, role, content, metadata, created_at 
                FROM chat.messages 
                WHERE session_id = %s 
                ORDER BY created_at ASC 
            """ 
             
            if limit: 
                query += f" LIMIT {limit}" 
             
            cursor.execute(query, (session_id,)) 
            messages = cursor.fetchall() 
             
            # Format interactions 
            interactions = [] 
            for msg in messages: 
                interactions.append({ 
                    "id": str(msg['id']), 
                    "role": msg['role'], 
                    "content": msg['content'], 
                    "metadata": msg['metadata'], 
                    "timestamp": msg['created_at'].isoformat() 
                }) 
             
            return { 
                "session_id": str(session_info['id']), 
                "user_id": session_info['user_id'], 
                "created_at": session_info['created_at'].isoformat(), 
                "interactions": interactions 
            } 
         
        finally: 
            cursor.close() 
     
    def get_conversation_context( 
        self, 
        user_id: str, 
        session_id: str, 
        max_messages: int = 10 
    ) -> List[Dict[str, str]]: 
        """ 
        Get recent conversation context for RAG (format suitable for LLM). 
         
        Args: 

            user_id: Unique identifier for the user 
            session_id: Session UUID 
            max_messages: Maximum number of recent messages to retrieve 
             
        Returns: 
            List of message dictionaries with 'role' and 'content' 
        """ 
        history = self.get_user_history(user_id, session_id, 
limit=max_messages) 
         
        if not history: 
            return [] 
         
        return [ 
            {"role": msg["role"], "content": msg["content"]} 
            for msg in history["interactions"] 
        ] 
     
    def get_user_sessions( 
        self, 
        user_id: str, 
        limit: int = 20 
    ) -> List[Dict[str, Any]]: 
        """ 
        Get all sessions for a user. 
         
        Args: 
            user_id: Unique identifier for the user 
            limit: Maximum number of sessions to retrieve 
             
        Returns: 
            List of session dictionaries 
        """ 
        conn = self._get_connection() 
        cursor = conn.cursor(cursor_factory=RealDictCursor) 
         
        try: 
            cursor.execute( 
                """ 
                SELECT  
                    s.id, 
                    s.user_id, 
                    s.created_at, 
                    COUNT(m.id) as message_count 
                FROM chat.sessions s 
                LEFT JOIN chat.messages m ON s.id = m.session_id 
                WHERE s.user_id = %s 
                GROUP BY s.id, s.user_id, s.created_at 

                ORDER BY s.created_at DESC 
                LIMIT %s 
                """, 
                (user_id, limit) 
            ) 
             
            sessions = cursor.fetchall() 
             
            return [ 
                { 
                    "session_id": str(session['id']), 
                    "user_id": session['user_id'], 
                    "created_at": session['created_at'].isoformat(), 
                    "message_count": session['message_count'] 
                } 
                for session in sessions 
            ] 
         
        finally: 
            cursor.close() 
     
    def delete_session(self, user_id: str, session_id: str) -> bool: 
        """ 
        Delete a chat session and all its messages. 
         
        Args: 
            user_id: Unique identifier for the user 
            session_id: Session UUID to delete 
             
        Returns: 
            bool: True if deleted, False if not found 
        """ 
        conn = self._get_connection() 
        cursor = conn.cursor() 
         
        try: 
            cursor.execute( 
                """ 
                DELETE FROM chat.sessions 
                WHERE id = %s AND user_id = %s 
                RETURNING id 
                """, 
                (session_id, user_id) 
            ) 
            deleted = cursor.fetchone() is not None 
            conn.commit() 
            return deleted 
         

        except Exception as e: 
            conn.rollback() 
            raise Exception(f"Error deleting session: {str(e)}") 
        finally: 
            cursor.close() 
     
    def close(self): 
        """Close database connection.""" 
        if self.connection and not self.connection.closed: 
            self.connection.close() 
     
    def __enter__(self): 
        """Context manager entry.""" 
        return self 
     
    def __exit__(self, exc_type, exc_val, exc_tb): 
        """Context manager exit.""" 
        self.close() 
 
