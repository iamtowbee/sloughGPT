import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import { store } from './store'

import Home from './components/Home.vue'
import Chat from './components/Chat.vue'
import Datasets from './components/Datasets.vue'
import Models from './components/Models.vue'
import Monitoring from './components/Monitoring.vue'

const routes = [
  { path: '/', component: Home },
  { path: '/chat', component: Chat },
  { path: '/datasets', component: Datasets },
  { path: '/models', component: Models },
  { path: '/monitoring', component: Monitoring }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

const app = createApp(App)
app.use(router)
app.use(store)
app.mount('#app')

export { app, router, store }