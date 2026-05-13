'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { usePipelineStore } from '@/store/usePipelineStore';
import { useEffect } from 'react';

import { API_BASE_URL } from '@/lib/config';

export function useStatus() {
  const { setStatus } = usePipelineStore();
  
  const query = useQuery({
    queryKey: ['status'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/status`);
      if (!res.ok) throw new Error('Failed to fetch status');
      const data = await res.json();
      return data;
    },
    refetchInterval: 5000,
    staleTime: 2000,
    refetchOnWindowFocus: false,
  });

  useEffect(() => {
    if (query.data) {
      setStatus(query.data);
    }
  }, [query.data, setStatus]);

  return query;
}

export function useReviewQueue() {
  return useQuery({
    queryKey: ['review-queue'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/review-queue`);
      if (!res.ok) throw new Error('Failed to fetch review queue');
      return res.json();
    },
    refetchInterval: 5000,
    staleTime: 3000,
    refetchOnWindowFocus: false,
  });
}

export function useReport() {
  return useQuery({
    queryKey: ['report'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/report`);
      if (!res.ok) throw new Error('Failed to fetch report');
      return res.json();
    },
    refetchInterval: 10000,
    staleTime: 5000,
    refetchOnWindowFocus: false,
  });
}

export function useReviewerDecisions() {
  return useQuery({
    queryKey: ['reviewer-decisions'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/reviewer/decisions`);
      if (!res.ok) throw new Error('Failed to fetch reviewer decisions');
      return res.json();
    },
    refetchInterval: 5000,
  });
}

export function useSubmitReview() {
  const queryClient = useQueryClient();
  const { setStatus } = usePipelineStore();
  
  return useMutation({
    mutationFn: async ({ txnId, decision }: { txnId: string, decision: 'ALLOW' | 'BLOCK' | 'SKIP' }) => {
      const res = await fetch(`${API_BASE_URL}/review/${txnId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision }),
      });
      if (!res.ok) throw new Error('Failed to submit review');
      return res.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['review-queue'] });
      queryClient.invalidateQueries({ queryKey: ['report'] });
      queryClient.invalidateQueries({ queryKey: ['reviewer-decisions'] });
      
      // Update UI with retraining status
      console.log("Model is retraining itself with your feedback...");
      
      fetch(`${API_BASE_URL}/status`)
        .then(res => res.json())
        .then(data => setStatus(data));
    },
  });
}

export function useRunPipeline() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (maxRows: number = 5000) => {
      console.log('Attempting to start pipeline...', { maxRows });
      const res = await fetch(`${API_BASE_URL}/run-pipeline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_rows: maxRows }),
      });
      if (!res.ok) {
        const error = await res.json();
        console.error('Pipeline start failed:', error);
        throw new Error(error.detail || 'Failed to start pipeline');
      }
      return res.json();
    },
    onSuccess: (data) => {
      console.log('Pipeline started successfully:', data);
      // Immediately update local store state so the button reflects 'Active' instantly
      usePipelineStore.getState().setStatus({
        pipeline_running: true,
        pipeline_done: false,
        total_processed: 0,
        review_queue: 0
      });
      queryClient.invalidateQueries({ queryKey: ['status'] });
    },
    onError: (error) => {
      console.error('Mutation error:', error);
    }
  });
}
