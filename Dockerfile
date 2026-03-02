FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=optional || npm i --omit=optional
COPY tsconfig.json ./
COPY src ./src
COPY coloranalysis.js ./coloranalysis.js
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY package*.json ./
RUN npm ci --omit=dev --omit=optional || npm i --omit=dev --omit=optional
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/coloranalysis.js ./coloranalysis.js
EXPOSE 9000
CMD ["node", "dist/server.js"]
