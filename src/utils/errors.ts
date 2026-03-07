export class ExternalServiceError extends Error {
  constructor(message: string, public readonly service: string) {
    super(message);
    this.name = "ExternalServiceError";
  }
}

export class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ValidationError";
  }
}

export class TimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TimeoutError";
  }
}
