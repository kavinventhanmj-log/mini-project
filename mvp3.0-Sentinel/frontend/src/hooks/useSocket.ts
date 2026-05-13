import { useEffect, useState, useRef } from 'react';
import { usePipelineStore } from '@/store/usePipelineStore';
import { WS_URL, API_BASE_URL } from '@/lib/config';

export interface Transaction {
  txn_num: number;
  transaction_id: string;
  user_id: string;
  amount: number;
  merchant: string;
  device: string;
  location: string;
  timestamp: string;
  risk_score: number;
  decision: 'ALLOW' | 'REVIEW' | 'BLOCK';
  shap: any[];
}

export function useSocket() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const { setStatus } = usePipelineStore();
  const socketRef = useRef<WebSocket | null>(null);
  
  // Use a ref to store incoming transactions to avoid excessive re-renders
  const pendingTransactions = useRef<Transaction[]>([]);
  const hasNewData = useRef(false);

  useEffect(() => {
    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === 'transaction') {
        pendingTransactions.current = [message.data, ...pendingTransactions.current].slice(0, 100);
        hasNewData.current = true;
      } else if (message.type === 'pipeline_complete') {
        fetch(`${API_BASE_URL}/status`)
          .then((res) => res.json())
          .then((data) => setStatus(data));
      }
    };

    // Periodically flush pending transactions to the state (every 100ms)
    // This batches updates and makes the UI much smoother
    const interval = setInterval(() => {
      if (hasNewData.current) {
        setTransactions([...pendingTransactions.current]);
        hasNewData.current = false;
      }
    }, 100);

    socket.onclose = () => {
      console.log('WebSocket disconnected');
      clearInterval(interval);
    };

    return () => {
      socket.close();
      clearInterval(interval);
    };
  }, [setStatus]);

  return { transactions };
}
