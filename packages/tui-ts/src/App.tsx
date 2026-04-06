import { SloughGPTClient } from '@sloughgpt/typescript-sdk';
import React, { useCallback, useEffect, useState } from 'react';
import { Box, Text, useApp, useInput } from 'ink';

type Props = {
  baseUrl: string;
};

export default function App({ baseUrl }: Props) {
  const { exit } = useApp();
  const [line1, setLine1] = useState('Loading…');
  const [line2, setLine2] = useState('');

  const refresh = useCallback(async () => {
    setLine1('Loading…');
    setLine2('');
    try {
      const client = new SloughGPTClient({ baseUrl });
      const h = await client.health();
      setLine1(`Health: ${h.status}`);
      setLine2(
        `model_loaded=${String(h.model_loaded)}  model_type=${h.model_type ?? 'n/a'}`
      );
    } catch (e: unknown) {
      setLine1('Health: error');
      setLine2(e instanceof Error ? e.message : String(e));
    }
  }, [baseUrl]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useInput((input, key) => {
    if (input === 'q' || key.escape) {
      exit();
    }
    if (input === 'r') {
      void refresh();
    }
  });

  return (
    <Box flexDirection="column" padding={1}>
      <Text bold color="cyan">
        SloughGPT TUI
      </Text>
      <Text dimColor>API {baseUrl}</Text>
      <Text>{line1}</Text>
      <Text>{line2}</Text>
      <Text dimColor>[r] refresh  [q] quit</Text>
    </Box>
  );
}
