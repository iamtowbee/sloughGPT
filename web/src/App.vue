<template>
  <div id="app">
    <div class="flex h-screen bg-gray-50">
      <!-- Sidebar -->
      <div class="w-64 bg-white shadow-lg">
        <div class="p-4 border-b">
          <h1 class="text-xl font-bold text-blue-600">SloughGPT</h1>
          <p class="text-sm text-gray-500">v2.0.0</p>
        </div>
        
        <nav class="p-4">
          <router-link 
            v-for="route in routes" 
            :key="route.path"
            :to="route.path"
            class="block px-4 py-2 rounded-md text-sm transition-colors mb-2
              hover:bg-blue-50 hover:text-blue-600
              text-gray-700 dark:text-gray-300"
            :class="{
              'bg-blue-600 text-white': $route.path === route.path
            }"
          >
            {{ route.path === '/' ? 'Dashboard' : route.path.slice(1).charAt(0).toUpperCase() + route.path.slice(2) }}
          </router-link>
        </nav>
        
        <div class="p-4 border-t">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs text-gray-500">Status</span>
            <Badge variant="secondary" size="sm" class="text-white">
              Online
            </Badge>
          </div>
          <div class="flex items-center justify-between text-xs text-gray-500">
            <span>API: Connected</span>
            <Badge variant="success" size="sm" class="text-white">
              ✓
            </Badge>
          </div>
        </div>
      </div>

      <!-- Main Content -->
      <div class="flex-1 overflow-auto">
        <router-view />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useStore } from './store'

const store = useStore()
const routes = [
  { path: '/', component: () => import('./components/Home.vue') },
  { path: '/chat', component: () => import('./components/Chat.vue') },
  { path: '/datasets', component: () => import('./components/Datasets.vue') },
  { path: '/models', component: () => import('./components/Models.vue') },
  { path: '/monitoring', component: () => import('./components/Monitoring.vue') }
]

const activeTab = computed(() => store.activeTab)
</script>