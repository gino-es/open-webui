<script lang="ts">
	import { getContext } from 'svelte';
	import Checkbox from '$lib/components/common/Checkbox.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import { marked } from 'marked';

	const i18n = getContext('i18n');

	export let responsesConfig: {
		enabled?: boolean;
		reasoning?: {
			enabled?: boolean;
			effort?: 'minimal' | 'low' | 'medium' | 'high';
			summary?: 'auto' | 'concise' | 'detailed';
		};
		background?: boolean;
		stream?: boolean;
	} = {
		enabled: false,
		reasoning: {
			enabled: false,
			effort: 'medium',
			summary: 'auto'
		},
		background: false,
		stream: false
	};

	const helpText = {
		enabled: $i18n.t('Enable special handling for /responses endpoint (for models like o3-pro, gpt-5)'),
		reasoning_enabled: $i18n.t('Enable reasoning output for supported models'),
		background: $i18n.t('Process requests in background (faster initial response, poll for completion)'),
		stream: $i18n.t('Enable streaming responses')
	};

	// Reactive statement to ensure reasoning is disabled when main config is disabled
	$: if (!responsesConfig.enabled) {
		responsesConfig.reasoning.enabled = false;
	}
</script>

<div>
	<div class="flex w-full justify-between mb-2">
		<div class="self-center text-sm font-semibold">{$i18n.t('Responses Configuration')}</div>
		<Tooltip content={marked.parse($i18n.t('Configuration for OpenAI /responses endpoint behavior'))}>
			<svg
				xmlns="http://www.w3.org/2000/svg"
				fill="none"
				viewBox="0 0 24 24"
				stroke-width="1.5"
				stroke="currentColor"
				class="w-4 h-4 text-gray-400"
			>
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 5.25h.008v.008H12v-.008Z"
				/>
			</svg>
		</Tooltip>
	</div>

	<div class="space-y-3 p-3 bg-gray-50 dark:bg-gray-950 rounded-lg">
		<!-- Main Enable Toggle -->
		<div class="flex items-center gap-3">
			<Checkbox
				state={responsesConfig.enabled ? 'checked' : 'unchecked'}
				on:change={(e) => {
					responsesConfig.enabled = e.detail === 'checked';
				}}
			/>
			<div class="text-sm">
				<Tooltip content={marked.parse(helpText.enabled)}>
					{$i18n.t('Enable Responses Endpoint')}
				</Tooltip>
			</div>
		</div>

		{#if responsesConfig.enabled}
			<div class="ml-6 space-y-3 border-l-2 border-gray-200 dark:border-gray-700 pl-4">
				<!-- Reasoning Configuration -->
				<div class="space-y-2">
					<div class="flex items-center gap-3">
						<Checkbox
							state={responsesConfig.reasoning.enabled ? 'checked' : 'unchecked'}
							on:change={(e) => {
								responsesConfig.reasoning.enabled = e.detail === 'checked';
							}}
						/>
						<div class="text-sm">
							<Tooltip content={marked.parse(helpText.reasoning_enabled)}>
								{$i18n.t('Enable Reasoning')}
							</Tooltip>
						</div>
					</div>

					{#if responsesConfig.reasoning.enabled}
						<div class="ml-6 space-y-2">
							<div class="flex items-center gap-3">
								<label class="text-xs font-medium text-gray-600 dark:text-gray-300 w-16">
									{$i18n.t('Effort')}:
								</label>
								<select
									bind:value={responsesConfig.reasoning.effort}
									class="text-xs bg-transparent border border-gray-300 dark:border-gray-600 rounded px-2 py-1"
								>
									<option value="minimal">{$i18n.t('Minimal')}</option>
									<option value="low">{$i18n.t('Low')}</option>
									<option value="medium">{$i18n.t('Medium')}</option>
									<option value="high">{$i18n.t('High')}</option>
								</select>
							</div>

							<div class="flex items-center gap-3">
								<label class="text-xs font-medium text-gray-600 dark:text-gray-300 w-16">
									{$i18n.t('Summary')}:
								</label>
								<select
									bind:value={responsesConfig.reasoning.summary}
									class="text-xs bg-transparent border border-gray-300 dark:border-gray-600 rounded px-2 py-1"
								>
									<option value="auto">{$i18n.t('Auto')}</option>
									<option value="concise">{$i18n.t('Concise')}</option>
									<option value="detailed">{$i18n.t('Detailed')}</option>
								</select>
							</div>
						</div>
					{/if}
				</div>

				<!-- Background Processing -->
				<div class="flex items-center gap-3">
					<Checkbox
						state={responsesConfig.background ? 'checked' : 'unchecked'}
						on:change={(e) => {
							responsesConfig.background = e.detail === 'checked';
						}}
					/>
					<div class="text-sm">
						<Tooltip content={marked.parse(helpText.background)}>
							{$i18n.t('Background Processing')}
						</Tooltip>
					</div>
				</div>

				<!-- Streaming -->
				<div class="flex items-center gap-3">
					<Checkbox
						state={responsesConfig.stream ? 'checked' : 'unchecked'}
						on:change={(e) => {
							responsesConfig.stream = e.detail === 'checked';
						}}
					/>
					<div class="text-sm">
						<Tooltip content={marked.parse(helpText.stream)}>
							{$i18n.t('Enable Streaming')}
						</Tooltip>
					</div>
				</div>
			</div>
		{/if}
	</div>
</div>
